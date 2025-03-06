"""
Our implementation builds upon the implentation of [1].
We made several modifications and additions, including:
(i) Introducing a transformer-based network architecture,
(ii) Incorporating analytic sampling of the bridge process and numerical simulation of SDEs in the spectral domain.
(iii) Resolution-free generation.

References:
[1] IDBM: https://github.com/stepelu/idbm-pytorch
"""

import warnings
import os
import torchvision.transforms.functional as F

import fire
import numpy as np
import torch as th
import wandb
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from models.transformer import OperatorTransformer
from dct import dct_2d, idct_2d

# DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")

data_path = os.path.expanduser("~/torch-data/")

# DDP
if th.cuda.is_available() and th.cuda.device_count() > 1:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = th.device(f"cuda:{rank}") if th.cuda.is_available() else th.device("cpu")
    parallel = True
else:
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
    rank = 0
    world_size = 1
    parallel = False

DBFS = "DBFS"


# Coarsening function
def coarsen_field(field, filter_sigma=2.0, downsample_factor=2, method="bilinear"):
    """
    Coarsen a field by smoothing and downsampling.

    Args:
        field (torch.Tensor): Input tensor of shape [B, C, H, W] or [C, H, W].
        filter_sigma (float): Standard deviation for Gaussian kernel.
        downsample_factor (int): Factor by which to downsample spatial dimensions.
        method (str): Interpolation method - 'bilinear' or 'bicubic'.

    Returns:
        torch.Tensor: Coarsened tensor.
    """
    if not isinstance(field, th.Tensor):
        raise ValueError("Field must be a torch tensor")

    # Handle input dimensions
    if field.dim() == 3:
        field = field.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
        squeeze = True
    elif field.dim() == 4:
        squeeze = False
    else:
        raise ValueError("Field must have 3 or 4 dimensions [C, H, W] or [B, C, H, W]")

    # Calculate kernel size: typically 4 times sigma, rounded to the nearest odd integer
    kernel_size = 2 * int(4 * filter_sigma + 0.5) + 1

    # Apply Gaussian smoothing
    smooth = F.gaussian_blur(
        field,
        kernel_size=(kernel_size, kernel_size),
        sigma=(filter_sigma, filter_sigma),
    )

    # Downsample using interpolation
    scale_factor = 1.0 / downsample_factor
    coarse = th.nn.functional.interpolate(
        smooth, scale_factor=scale_factor, mode=method, align_corners=False
    )

    if squeeze:
        coarse = coarse.squeeze(0)  # Remove batch dimension if it was added

    return coarse


# Routines -----------------------------------------------------------------------------


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def a_(s, t, a_k):
    #  t > s
    return th.exp(-a_k * (t - s))


def a_2(s, t, a_k):
    #  t > s
    return th.exp(-2 * a_k * (t - s))


def v_(s, t, a_k):
    # t > s
    coeff = 1 / (2 * a_k)
    return coeff * (1 - a_2(s, t, a_k))


def sample_bridge(x_0, x_1, t, energy):
    t = t[:, None, None, None]
    t_start = th.zeros_like(t)
    t_end = th.ones_like(t)

    _, _, h, w = x_0.shape

    freqs = np.pi * th.linspace(0, h - 1, h) / h
    freq = (freqs[:, None] ** 2 + freqs[None, :] ** 2).to(x_0.device)
    frequencies_squared = freq + 1.0

    a_k = frequencies_squared[None, None]
    Sigma_k = th.pow(a_k, -0.02) / (energy**2)

    denominator = v_(t_start, t, a_k) * a_2(t, t_end, a_k) + v_(t, t_end, a_k)
    coeff_1 = (v_(t, t_end, a_k) * a_(t_start, t, a_k)) / denominator
    coeff_2 = (v_(t_start, t, a_k) * a_(t, t_end, a_k)) / denominator
    coeff_3 = (v_(t_start, t, a_k) * v_(t, t_end, a_k)) / denominator

    x_0 = dct_2d(x_0, norm="ortho")
    x_1 = dct_2d(x_1, norm="ortho")

    mean_t = coeff_1 * x_0 + coeff_2 * x_1
    var_t = coeff_3 * Sigma_k
    z_t = dct_2d(th.randn_like(x_0), norm="ortho")

    x_t = mean_t + th.sqrt(var_t) * z_t
    x_t = idct_2d(x_t, norm="ortho")

    return x_t


def dbfs_target(x_t, x_1, t):
    return x_1


def fwd_target(x_t, x_1, t=None):
    """Forward drift target for Brownian bridge (currently omitted the coefficients)
    # ! for actual spectral domain implementation, we need to include the (time-dependent) coefficients
    # ! in essence, we will need to do DCT to x_t and x_1, then multiply by the coefficients, then do IDCT
    # ! but the paper in DBFS framed it as "preconditioning" and did not include them
    Target drift for BM² algorithm in the forward direction.

    Note: loss_f_t = (target_f_t - prediction_f_t) ** 2
    """

    if t is None:
        return x_1
    # Adapt the shape of t for broadcasting
    t = t[:, None, None, None]
    # This means that given a t, in the forward direction, given a sampled x_t, this
    # target is the expected value of x_1 given x_t and t as this is the Brownian bridge
    # signifying interpolation between x_0 and x_1 at time t
    return (x_1 - x_t) / (1.0 - t)


def bwd_target(x_t, x_0, t=None):
    """Backward drift target for Brownian bridge in  the spectral domain (currently omitted the coefficients)
    # ! for actual spectral domain implementation, we need to include the (time-dependent) coefficients
    # ! in essence, we will need to do DCT to x_t and x_0, then multiply by the coefficients, then do IDCT
    # ! but the paper in DBFS framed it as "preconditioning" and did not include them
    Target drift for BM² algorithm in the backward direction.
    """

    if t is None:
        return x_0
    # Adapt the shape of t for broadcasting
    t = t[:, None, None, None]
    # This means that given a t, in the backward direction, given a sampled x_t, this
    # target is the expected value of x_0 given x_t and t as this is the Brownian bridge
    # signifying interpolation between x_1 and x_0 at time t
    return (x_0 - x_t) / t


def consistency_target(x_t, x_0, x_1, t):
    """Consistency target for BM² algorithm.
    Implements the γ_01 term from equation (8) in the paper.
    """
    # Adapt the shape of t for broadcasting
    t = t[:, None, None, None]
    # Implementation of equation (8) from the paper
    return (x_0 - x_t) / t + (x_1 - x_t) / (1.0 - t)


def euler_discretization(x, xp, nn, energy, chunk_size=128):
    # x has shape [T+1, cache_batch_dim, C, H, W]
    T = x.shape[0] - 1  # number of discretization steps
    B = x.shape[1]  # B = cache_batch_dim
    dt = th.full(size=(B,), fill_value=1.0 / T, device=x.device)
    drift_norms = 0.0

    _, b, c, h, w = x.shape

    freqs = np.pi * th.linspace(0, h - 1, h) / h
    freq = (freqs[:, None] ** 2 + freqs[None, :] ** 2).to(x.device)
    frequencies_squared = freq + 1.0
    a_k = frequencies_squared[None, None]
    sigma_k = th.pow(a_k, -0.01) / energy

    for i in range(1, T + 1):
        t = dt * (i - 1)
        # Instead of processing the full cache at once, process in chunks:
        alpha_chunks = []
        # Split the batch dimension into sub-batches of size chunk_size:
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            # Process a small chunk through the neural network:
            alpha_chunk = nn(x[i - 1][start:end], t[start:end])
            alpha_chunks.append(alpha_chunk)
        # Concatenate the results along the batch dimension:
        alpha_t = th.cat(alpha_chunks, dim=0)
        drift_norms = drift_norms + th.mean(alpha_t.view(B, -1) ** 2, dim=1)

        # Continue with the rest of Euler discretization using alpha_t,
        # Convert alpha_t to spectral domain:
        alpha_t = dct_2d(alpha_t, norm="ortho")
        # Convert x[i-1] to spectral domain:
        x_i = dct_2d(x[i - 1], norm="ortho")
        # (Compute drift, diffusion, etc.)
        t_ = t[:, None, None, None]
        t_end = th.ones_like(t_)
        xp_coeff = alpha_t
        xp[i - 1] = idct_2d(xp_coeff, norm="ortho")
        control = (a_(t_, t_end, a_k) * alpha_t - a_2(t_, t_end, a_k) * x_i) / v_(
            t_, t_end, a_k
        )
        drift_t = (-a_k * x_i + control) * dt[:, None, None, None]
        eps_t = dct_2d(th.randn_like(x[i - 1]), norm="ortho")

        if i == T:
            diffusion_t = 0
        else:
            diffusion_t = sigma_k * th.sqrt(dt[:, None, None, None]) * eps_t

        # Update x[i] accordingly
        x[i] = idct_2d(x_i + drift_t + diffusion_t, norm="ortho")

        # Optionally compute and accumulate drift_norms as needed.
    drift_norms = drift_norms / T
    return drift_norms.cpu()


# Data ---------------------------------------------------------------------------------
class FieldsDataset(Dataset):
    """Dataset for 2D fields (coarse or fine)"""

    def __init__(self, data_path, train=True, transform=None):
        # Load data from NumPy file
        data = np.load(data_path)
        n_samples = len(data)
        n_train = int(0.8 * n_samples)

        # Split into train/test sets
        if train:
            self.data = data[:n_train]
        else:
            self.data = data[n_train:]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        field = self.data[idx]
        if self.transform:
            field = self.transform(field)
        # Ensure field has shape (1, H, W)
        assert field.dim() == 3, f"Expected field to be 3D, got {field.dim()}D"
        return field


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.epoch_iterator)
        except StopIteration:
            self.epoch_iterator = super().__iter__()
            batch = next(self.epoch_iterator)
        return batch


def train_iter(data, batch_dim, sampler=None):
    return iter(
        InfiniteDataLoader(
            dataset=data,
            batch_size=batch_dim,
            num_workers=4 * world_size,
            pin_memory=True,
            # shuffle=True,
            drop_last=True,
            sampler=sampler,
        )
    )


def test_loader(data, batch_dim, sampler=None):
    return DataLoader(
        dataset=data,
        batch_size=batch_dim,
        num_workers=4 * world_size,
        pin_memory=True,
        # shuffle=False,
        drop_last=True,
        sampler=sampler,
    )


def resample_indices(from_n, to_n):
    # Equi spaced resampling, first and last element always included.
    return np.round(np.linspace(0, from_n - 1, num=to_n)).astype(int)


def image_grid(x, normalize=False, n=5):
    img = x[: n**2].cpu()
    img = make_grid(img, nrow=n, normalize=normalize, scale_each=normalize)
    img = wandb.Image(img)
    return img


# For fixed permutations of test sets:
rng = np.random.default_rng(seed=0x87351080E25CB0FAD77A44A3BE03B491)


class ToTensorCustom:
    def __call__(self, array):
        tensor = th.from_numpy(array).float()
        # Add channel dimension if missing
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # Now tensor shape is (1, H, W)
        return tensor


class DatasetNormalization:
    def __init__(self, train_data_path):
        # Load training data
        train_data = np.load(train_data_path)
        self.mean = train_data.mean()
        self.std = train_data.std()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = th.from_numpy(x).float()
        # Normalize to [-1,1] using training statistics
        return (x - self.mean) / self.std


# Add denormalization class
class DatasetDenormalization:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x * self.std + self.mean


# Create normalizers using only training data
coarse_normalizer = DatasetNormalization("../Data/coarse_grf_10k.npy")
fine_normalizer = DatasetNormalization("../Data/fine_grf_10k.npy")

# Normalization transform
transform_coarse = transforms.Compose([ToTensorCustom(), coarse_normalizer])

transform_fine = transforms.Compose([ToTensorCustom(), fine_normalizer])

# Load datasets for coarse (source) and fine (target) fields
tr_data_0 = FieldsDataset(
    data_path="../Data/coarse_grf_10k.npy",
    train=True,
    transform=transform_coarse,
)
te_data_0 = FieldsDataset(
    data_path="../Data/coarse_grf_10k.npy",
    train=False,
    transform=transform_coarse,
)
tr_data_1 = FieldsDataset(
    data_path="../Data/fine_grf_10k.npy",
    train=True,
    transform=transform_fine,
)
te_data_1 = FieldsDataset(
    data_path="../Data/fine_grf_10k.npy",
    train=False,
    transform=transform_fine,
)
# Upsample the coarse fields to match the size of fine fields
import torch.nn.functional as Func


def upsample(field):
    field = Func.interpolate(
        field.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
    ).squeeze(0)
    return field


upsample_transform = transforms.Compose([transforms.Lambda(upsample)])

# Update the transforms for the coarse field datasets
tr_data_0.transform = transforms.Compose([tr_data_0.transform, upsample_transform])
te_data_0.transform = transforms.Compose([te_data_0.transform, upsample_transform])

# Update the transforms for the coarse field datasets
tr_data_0.transform = transforms.Compose([tr_data_0.transform, upsample_transform])
te_data_0.transform = transforms.Compose([te_data_0.transform, upsample_transform])

te_data_0 = Subset(te_data_0, rng.permutation(len(te_data_0)))
te_data_1 = Subset(te_data_1, rng.permutation(len(te_data_1)))

# DDP samplers
tr_spl_0 = DistributedSampler(tr_data_0, shuffle=True) if parallel else None
tr_spl_1 = DistributedSampler(tr_data_1, shuffle=True) if parallel else None
te_spl_0 = DistributedSampler(te_data_0, shuffle=False) if parallel else None
te_spl_1 = DistributedSampler(te_data_1, shuffle=False) if parallel else None

# NN Model -----------------------------------------------------------------------------


def init_nn():
    return OperatorTransformer(
        in_channel=2,
        out_channel=1,
        latent_dim=256,
        pos_dim=256,
        num_heads=4,
        depth_enc=6,
        depth_dec=2,
        scale=1,
        self_per_cross_attn=1,
        height=256,  # to match the fine fields
    )


# Modified version for BM² that supports both forward and backward directions
def init_bm2_nn():
    """Initialize a neural network for BM² that can handle both forward and backward directions.
    This version uses the dedicated BM2Transformer model with separate output heads.
    """
    from dbfs.models.bm2_transformer import init_bm2_model

    return init_bm2_model(
        in_channel=2,
        out_channel=1,
        pos_dim=256,
        latent_dim=256,
        num_heads=4,
        depth_enc=6,
        depth_dec=2,
        scale=1,
        self_per_cross_attn=1,
        height=256,  # to match the fine fields
        dim=2,  # 2D data
    )


class EMAHelper:
    # Simplified from https://github.com/ermongroup/ddim/blob/main/models/ema.py:
    def __init__(self, module, mu=0.999, device=None):
        self.module = module.module if isinstance(module, DDP) else module  # DDP
        self.mu = mu
        self.device = device
        self.shadow = {}
        # Register:
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self):
        locs = self.module.locals
        module_copy = type(self.module)(*locs).to(self.device)
        module_copy.load_state_dict(self.module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# Run ----------------------------------------------------------------------------------


def run(
    method=DBFS,
    sigma=1.0,
    intOp_scale_factor=0.1,  # Add intOp_scale_factor parameter with default value
    iterations=60,
    training_steps=5000,
    discretization_steps=30,
    batch_dim=32,
    learning_rate=1e-4,
    grad_max_norm=1.0,
    ema_decay=0.999,
    cache_steps=50,
    cache_batch_dim=32,
    test_steps=5000,
    test_batch_dim=16,
    loss_log_steps=100,
    imge_log_steps=1000,
    load=False,
    use_bm2=True,  # Flag to enable BM² algorithm
    two_stage_training=True,  # Enable two-stage training as described in the paper
    consistency_loss_weight=1.0,  # Weight for the consistency loss term
    t_epsilon=1e-5,  # Avoid singularities by sampling t from (epsilon, 1-epsilon)
):
    config = locals()
    assert isinstance(sigma, float) and sigma >= 0
    assert isinstance(learning_rate, float) and learning_rate > 0
    assert isinstance(grad_max_norm, float) and grad_max_norm >= 0
    assert method in [DBFS]

    console = Console(log_path=False)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        TextColumn("•"),
        MofNCompleteColumn(),
        console=console,
        speed_estimate_period=60 * 5,
    )
    iteration_t = progress.add_task("iteration", total=iterations)

    # Device setup:
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup data:
    tr_loader_0 = train_iter(tr_data_0, batch_dim)
    tr_iter_0 = iter(tr_loader_0)
    tr_cache_loader_0 = InfiniteDataLoader(
        tr_data_0, batch_size=cache_batch_dim, shuffle=True
    )
    tr_cache_iter_0 = iter(tr_cache_loader_0)
    te_loader_0 = test_loader(te_data_0, test_batch_dim)

    tr_loader_1 = train_iter(tr_data_1, batch_dim)
    tr_iter_1 = iter(tr_loader_1)
    tr_cache_loader_1 = InfiniteDataLoader(
        tr_data_1, batch_size=cache_batch_dim, shuffle=True
    )
    tr_cache_iter_1 = iter(tr_cache_loader_1)
    te_loader_1 = test_loader(te_data_1, test_batch_dim)

    # Precompute discretization times:
    t_T = 1.0
    ts = th.linspace(0, t_T, discretization_steps + 1, device=device)
    ts_idx = th.round(ts * discretization_steps).long()
    ts_idx[-1] = discretization_steps

    # For caching, need to store endpoints of paths:
    s_path = th.zeros(
        discretization_steps + 1, cache_batch_dim, 1, 256, 256, device=device
    )  # [T+1, cache_batch_dim, C, H, W]
    p_path = th.zeros_like(s_path)  # Placeholder for debugging

    if use_bm2:
        # For BM², we use a single neural network for both forward and backward drift
        nn = init_bm2_nn().to(device)
        optim = th.optim.Adam(nn.parameters(), lr=learning_rate)

        # EMA helper for the model parameters
        ema = EMAHelper(nn, mu=ema_decay, device=device)

        # Sample model used for generating paths (with EMA parameters)
        sample_nn = ema.ema_copy()

        step = 0
        start_iteration = 0
        if load:
            # Load checkpoint if available
            start_iteration, step = restore_checkpoint(
                {"nn": nn, "optim": optim, "ema": ema, "sample_nn": sample_nn},
                console,
            )
            if rank == 0:
                console.log(
                    f"Loaded checkpoint: iteration {start_iteration}, step {step}"
                )

        step_t = progress.add_task("step", total=iterations * training_steps)

        scaler = th.cuda.amp.GradScaler(enabled=True)

        if rank == 0:
            progress.start()

        # Two-stage training as described in the BM² paper
        if two_stage_training:
            # First stage: Train independent Brownian motion transport in both directions
            console.log("Starting first stage: Independent BM transport")
            for stage1_step in range(1, training_steps + 1):
                progress.update(step_t, completed=stage1_step)
                optim.zero_grad()

                # Sample from the marginal distributions
                x_0 = next(tr_iter_0).to(device)
                x_1 = next(tr_iter_1).to(device)

                # Sample t avoiding singularities
                t = (
                    th.rand(size=(batch_dim,), device=device) * (1.0 - 2 * t_epsilon)
                    + t_epsilon
                )

                # Forward pass
                x_t_forward = sample_bridge(x_0, x_1, t, sigma)
                fwd_target_t = fwd_target(x_t_forward, x_1, t)

                # Backward pass
                x_t_backward = sample_bridge(x_0, x_1, t, sigma)
                bwd_target_t = bwd_target(x_t_backward, x_0, t)

                with th.autocast(device_type="cuda", dtype=th.float16, enabled=True):
                    # Predict both forward and backward drifts with the same network
                    # The network can distinguish between forward and backward based on a direction flag
                    fwd_pred = nn(x_t_forward, t, direction="fwd")
                    bwd_pred = nn(x_t_backward, t, direction="bwd")

                    # Independent BM loss
                    fwd_losses = (fwd_target_t - fwd_pred) ** 2
                    bwd_losses = (bwd_target_t - bwd_pred) ** 2

                    fwd_losses = th.mean(
                        fwd_losses.reshape(fwd_losses.shape[0], -1), dim=1
                    )
                    bwd_losses = th.mean(
                        bwd_losses.reshape(bwd_losses.shape[0], -1), dim=1
                    )

                    loss = th.mean(fwd_losses + bwd_losses)

                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = th.nn.utils.clip_grad_norm_(nn.parameters(), grad_max_norm)
                scaler.step(optim)
                scaler.update()
                ema.update()

                if stage1_step % loss_log_steps == 0:
                    console.log(
                        f"Stage 1 Step {stage1_step}: loss {loss:.4f}, grad {grad_norm:.4f}"
                    )

            console.log("Completed first stage training")

        # BM² training loop
        console.log("Starting BM² training")

        # Create a cache for endpoints
        fwd_cache_x0 = []
        fwd_cache_x1 = []
        bwd_cache_x0 = []
        bwd_cache_x1 = []

        # Initialize caches
        console.log("Initializing endpoint caches...")
        with th.no_grad():
            # Create forward path endpoints
            for _ in range(cache_batch_dim // batch_dim):
                x_0 = next(tr_iter_0).to(device)
                # Generate x_1 by simulating the forward process
                ema.ema(sample_nn)
                # We need to simulate the forward process to get x_1
                # This is a simplified version - in practice, you'd use euler_discretization
                t_ones = th.ones(x_0.shape[0], device=device)
                x_1 = sample_nn(x_0, t_ones, direction="fwd")
                fwd_cache_x0.append(x_0.detach())
                fwd_cache_x1.append(x_1.detach())

            # Create backward path endpoints
            for _ in range(cache_batch_dim // batch_dim):
                x_1 = next(tr_iter_1).to(device)
                # Generate x_0 by simulating the backward process
                t_zeros = th.zeros(x_1.shape[0], device=device)
                x_0 = sample_nn(x_1, t_zeros, direction="bwd")
                bwd_cache_x0.append(x_0.detach())
                bwd_cache_x1.append(x_1.detach())

        # Convert lists to tensors
        fwd_cache_x0 = th.cat(fwd_cache_x0, dim=0)
        fwd_cache_x1 = th.cat(fwd_cache_x1, dim=0)
        bwd_cache_x0 = th.cat(bwd_cache_x0, dim=0)
        bwd_cache_x1 = th.cat(bwd_cache_x1, dim=0)

        for iteration in range(start_iteration + 1, iterations + 1):
            if rank == 0:
                console.log(f"BM² iteration {iteration}: {step}")
            if rank == 0:
                progress.update(iteration_t, completed=iteration)

            # BM² training loop
            for step in range(step + 1, step + training_steps + 1):
                progress.update(step_t, completed=step)
                optim.zero_grad()

                # Refresh cache periodically
                if (step - 1) % cache_steps == 0:
                    with th.no_grad():
                        # Update sample_nn with latest EMA parameters
                        ema.ema(sample_nn)

                        # Update forward path endpoints
                        for i in range(0, cache_batch_dim, batch_dim):
                            end = min(i + batch_dim, cache_batch_dim)
                            batch_size = end - i
                            x_0 = next(tr_iter_0).to(device)[:batch_size]
                            # Generate x_1 by simulating the forward process
                            t_ones = th.ones(batch_size, device=device)
                            x_1 = sample_nn(x_0, t_ones, direction="fwd")
                            fwd_cache_x0[i:end] = x_0.detach()
                            fwd_cache_x1[i:end] = x_1.detach()

                        # Update backward path endpoints
                        for i in range(0, cache_batch_dim, batch_dim):
                            end = min(i + batch_dim, cache_batch_dim)
                            batch_size = end - i
                            x_1 = next(tr_iter_1).to(device)[:batch_size]
                            # Generate x_0 by simulating the backward process
                            t_zeros = th.zeros(batch_size, device=device)
                            x_0 = sample_nn(x_1, t_zeros, direction="bwd")
                            bwd_cache_x0[i:end] = x_0.detach()
                            bwd_cache_x1[i:end] = x_1.detach()

                # Randomly select samples from cache
                fwd_idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
                bwd_idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]

                # Get forward path endpoints
                f_0 = fwd_cache_x0[fwd_idx]
                f_1 = fwd_cache_x1[fwd_idx]

                # Get backward path endpoints
                b_0 = bwd_cache_x0[bwd_idx]
                b_1 = bwd_cache_x1[bwd_idx]

                # Sample t avoiding singularities
                t = (
                    th.rand(size=(batch_dim,), device=device) * (1.0 - 2 * t_epsilon)
                    + t_epsilon
                )

                # Sample bridge points
                pi_f_t = sample_bridge(f_0, f_1, t, sigma)
                pi_b_t = sample_bridge(b_0, b_1, t, sigma)

                # Compute targets and model predictions
                with th.autocast(device_type="cuda", dtype=th.float16, enabled=True):
                    # Forward targets and predictions
                    target_f_t = fwd_target(pi_b_t, b_1, t)
                    prediction_f_t = nn(pi_b_t, t, direction="fwd")

                    # Backward targets and predictions
                    target_b_t = bwd_target(pi_f_t, f_0, t)
                    prediction_b_t = nn(pi_f_t, t, direction="bwd")

                    # Compute the BM² coupled loss
                    loss_f_t = (target_f_t - prediction_f_t) ** 2
                    loss_b_t = (target_b_t - prediction_b_t) ** 2

                    loss_f_t = th.mean(loss_f_t.reshape(loss_f_t.shape[0], -1), dim=1)
                    loss_b_t = th.mean(loss_b_t.reshape(loss_b_t.shape[0], -1), dim=1)

                    loss = th.mean(loss_f_t + loss_b_t)

                    # ! NOTE: Currently not adding consistency loss as it is not mentioned in the paper
                    # ! Also because we need to reformulate it in the spectral domain, which is non-trivial for now.
                    # # Optionally add consistency loss
                    # if consistency_loss_weight > 0:
                    #     # Sample points from the mixture of forward and backward processes
                    #     if th.rand(1).item() < 0.5:
                    #         x_t_mix = pi_f_t
                    #         x_0_mix = f_0
                    #         x_1_mix = f_1
                    #     else:
                    #         x_t_mix = pi_b_t
                    #         x_0_mix = b_0
                    #         x_1_mix = b_1

                    #     # Compute consistency target
                    #     gamma_target = consistency_target(x_t_mix, x_0_mix, x_1_mix, t)

                    #     # Compute model predictions (forward and backward)
                    #     fwd_pred_mix = nn(x_t_mix, t, direction="fwd")
                    #     bwd_pred_mix = nn(x_t_mix, t, direction="bwd")

                    #     # Compute consistency loss
                    #     consistency_loss = (
                    #         fwd_pred_mix + bwd_pred_mix - gamma_target
                    #     ) ** 2
                    #     consistency_loss = th.mean(
                    #         consistency_loss.reshape(consistency_loss.shape[0], -1),
                    #         dim=1,
                    #     )

                    #     # Add weighted consistency loss
                    #     loss = loss + consistency_loss_weight * th.mean(
                    #         consistency_loss
                    #     )

                # Backpropagation and optimization
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = th.nn.utils.clip_grad_norm_(nn.parameters(), grad_max_norm)
                scaler.step(optim)
                scaler.update()
                ema.update()

                # Logging
                if step % loss_log_steps == 0 and rank == 0:
                    console.log(
                        f"BM² Step {step}: loss {loss:.4f}, grad {grad_norm:.4f}"
                    )

                # Checkpointing
                if step % imge_log_steps == 0 and rank == 0:
                    save_checkpoint(
                        {"nn": nn, "optim": optim, "ema": ema, "sample_nn": sample_nn},
                        iteration,
                        console,
                    )

                # Evaluation using current model
                if step % (10 * loss_log_steps) == 0 and rank == 0:
                    # Update sample_nn with latest EMA parameters
                    ema.ema(sample_nn)

                    # Evaluate on test set
                    with th.no_grad():
                        test_loss_fwd = 0.0
                        test_loss_bwd = 0.0
                        num_batches = 0

                        for x_0_test in te_loader_0:
                            x_0_test = x_0_test.to(device)
                            x_1_test = next(iter(te_loader_1)).to(device)
                            t_test = (
                                th.rand(size=(x_0_test.shape[0],), device=device)
                                * (1.0 - 2 * t_epsilon)
                                + t_epsilon
                            )

                            # Sample bridge points
                            pi_f_t_test = sample_bridge(
                                x_0_test, x_1_test, t_test, sigma
                            )
                            pi_b_t_test = sample_bridge(
                                x_0_test, x_1_test, t_test, sigma
                            )

                            # Compute targets
                            target_f_t_test = fwd_target(pi_b_t_test, x_1_test, t_test)
                            target_b_t_test = bwd_target(pi_f_t_test, x_0_test, t_test)

                            # Model predictions
                            pred_f_t_test = sample_nn(
                                pi_b_t_test, t_test, direction="fwd"
                            )
                            pred_b_t_test = sample_nn(
                                pi_f_t_test, t_test, direction="bwd"
                            )

                            # Compute test losses
                            loss_f_test = th.mean(
                                ((target_f_t_test - pred_f_t_test) ** 2).reshape(
                                    x_0_test.shape[0], -1
                                )
                            )
                            loss_b_test = th.mean(
                                ((target_b_t_test - pred_b_t_test) ** 2).reshape(
                                    x_0_test.shape[0], -1
                                )
                            )

                            test_loss_fwd += loss_f_test.item()
                            test_loss_bwd += loss_b_test.item()
                            num_batches += 1

                        test_loss_fwd /= num_batches
                        test_loss_bwd /= num_batches

                        console.log(
                            f"Test loss - Forward: {test_loss_fwd:.4f}, Backward: {test_loss_bwd:.4f}"
                        )

        return nn

    else:
        # Original DBFS code
        bwd_nn = init_nn().to(device)
        fwd_nn = init_nn().to(device)
        # DDP
        if parallel:
            bwd_nn = DDP(bwd_nn, device_ids=[rank])
            fwd_nn = DDP(fwd_nn, device_ids=[rank])

        if rank == 0:
            console.log(f"# param of bwd nn: {count_parameters(bwd_nn)}")
            console.log(f"# param of fwd nn: {count_parameters(fwd_nn)}")

        bwd_ema = EMAHelper(bwd_nn, ema_decay, device)
        fwd_ema = EMAHelper(fwd_nn, ema_decay, device)
        bwd_sample_nn = bwd_ema.ema_copy()
        fwd_sample_nn = fwd_ema.ema_copy()

        bwd_nn.train()
        fwd_nn.train()
        bwd_sample_nn.eval()
        fwd_sample_nn.eval()

        bwd_optim = th.optim.Adam(bwd_nn.parameters(), lr=learning_rate)
        fwd_optim = th.optim.Adam(fwd_nn.parameters(), lr=learning_rate)

        saves = [bwd_nn, bwd_ema, bwd_optim, fwd_nn, fwd_ema, fwd_optim]

        start_iteration = 0
        step = 0
        if load is True:
            start_iteration = restore_checkpoint(saves, console)

            bwd_sample_nn = bwd_ema.ema_copy()
            fwd_sample_nn = fwd_ema.ema_copy()
            bwd_nn.train()
            fwd_nn.train()
            bwd_sample_nn.eval()
            fwd_sample_nn.eval()

            step = start_iteration * training_steps

        if rank == 0:
            console.log(f"Training Start from Iteration : {start_iteration}")

        dt = 1.0 / discretization_steps
        t_T = 1.0 - dt * 0.5

        # Update image dimensions from 32x32 to 32x32
        s_path = th.zeros(
            size=(discretization_steps + 1,) + (cache_batch_dim, 1, 32, 32),
            device=device,
        )  # i: 0, ..., discretization_steps; t: 0, dt, ..., 1.0.
        p_path = th.zeros(
            size=(discretization_steps,) + (cache_batch_dim, 1, 32, 32), device=device
        )  # i: 0, ..., discretization_steps - 1; t: 0, dt, ..., 1.0 - dt.

        scaler = th.cuda.amp.GradScaler(enabled=True)

        if rank == 0:
            progress.start()
        for iteration in range(start_iteration + 1, iterations + 1):
            if rank == 0:
                console.log(f"iteration {iteration}: {step}")
            if rank == 0:
                progress.update(iteration_t, completed=iteration)
            # Setup:
            if (iteration % 2) != 0:
                # Odd iteration => bwd.
                direction = "bwd"
                nn = bwd_nn
                ema = bwd_ema
                sample_nn = bwd_sample_nn
                optim = bwd_optim
                te_loader_x_0 = te_loader_1
                te_loader_x_1 = te_loader_0

                def sample_dbfs_coupling(step):
                    if iteration == 1:
                        # Independent coupling:
                        x_0 = next(tr_iter_1).to(device)
                        x_1 = next(tr_iter_0).to(device)
                    else:
                        with th.no_grad():
                            if (step - 1) % cache_steps == 0:
                                if rank == 0:
                                    console.log(f"cache update: {step}")
                                # Simulate previously inferred SDE:
                                s_path[0] = next(tr_cache_iter_0).to(device)

                                euler_discretization(
                                    s_path, p_path, fwd_sample_nn, sigma
                                )
                            # Random selection:
                            idx = th.randperm(cache_batch_dim, device=device)[
                                :batch_dim
                            ]
                            # Reverse path:
                            x_0, x_1 = s_path[-1, idx], s_path[0, idx]
                    return x_0, x_1

            else:
                # Even iteration => fwd.
                direction = "fwd"
                nn = fwd_nn
                ema = fwd_ema
                sample_nn = fwd_sample_nn
                optim = fwd_optim
                te_loader_x_0 = te_loader_0
                te_loader_x_1 = te_loader_1

                def sample_dbfs_coupling(step):
                    with th.no_grad():
                        if (step - 1) % cache_steps == 0:
                            if rank == 0:
                                console.log(f"cache update: {step}")
                            # Simulate previously inferred SDE:
                            s_path[0] = next(tr_cache_iter_1).to(device)

                            euler_discretization(s_path, p_path, bwd_sample_nn, sigma)
                        # Random selection:
                        idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
                        # Reverse path:
                        x_0, x_1 = s_path[-1, idx], s_path[0, idx]
                    return x_0, x_1

            for step in range(step + 1, step + training_steps + 1):
                progress.update(step_t, completed=step)
                optim.zero_grad()

                # for backward direction, x_1 is the coarse field (not from euler discretization), x_0 is the fine field (from euler discretization)
                # for forward direction, x_0 is the coarse field (from euler discretization), x_1 is the fine field (not from euler discretization)
                x_0, x_1 = sample_dbfs_coupling(step)
                # print("Shape of x_0: ", x_0.shape)
                # print("Shape of x_1: ", x_1.shape)
                t = th.rand(size=(batch_dim,), device=device) * t_T
                x_t = sample_bridge(x_0, x_1, t, sigma)
                # print("Shape of x_t: ", x_t.shape)
                target_t = dbfs_target(x_t, x_1, t)

                with th.autocast(device_type="cuda", dtype=th.float16, enabled=True):
                    alpha_t = nn(x_t, t)
                    losses = (target_t - alpha_t) ** 2
                    losses = th.mean(losses.reshape(losses.shape[0], -1), dim=1)

                    if direction == "bwd":
                        # target_t in backward direction is the coarse field
                        # target_t is not obtained from the forward process, it is from the cache
                        # x_0 in backward direction is the fine field
                        # x_0 is obtained from euler discretization of the forward process under the control \alpha_t(;\theta)
                        losses_intOperator = (
                            target_t - coarsen_field(x_0, downsample_factor=1)
                        ) ** 2

                        losses_intOperator = th.mean(
                            losses_intOperator.reshape(losses_intOperator.shape[0], -1),
                            dim=1,
                        )

                        losses = losses + intOp_scale_factor * losses_intOperator

                    loss = th.mean(losses)

                scaler.scale(loss).backward()

                scaler.unscale_(optim)
                if grad_max_norm > 0:
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        nn.parameters(), grad_max_norm
                    )

                scaler.step(optim)
                scaler.update()
                ema.update()

                if step % test_steps == 0:
                    if rank == 0:
                        console.log(f"test: {step}")
                    ema.ema(sample_nn)

                # Save per each iteration
                saves = [bwd_nn, bwd_ema, bwd_optim, fwd_nn, fwd_ema, fwd_optim]
                save_checkpoint(saves, iteration, console)

                with th.no_grad():
                    te_s_path = th.zeros(
                        size=(discretization_steps + 1,) + (test_batch_dim, 1, 32, 32),
                        device=device,
                    )
                    te_p_path = th.zeros(
                        size=(discretization_steps,) + (test_batch_dim, 1, 32, 32),
                        device=device,
                    )
                    # Physical field so use L2 loss
                    l2_losses = []

                    drift_norm = []
                    # console.log("Drift norm computation")
                    # console.log("te_loader_x_0: ", len(te_loader_x_0))
                    # console.log("te_loader_x_1: ", len(te_loader_x_1))

                    for te_x_0, te_x_1 in zip(te_loader_x_0, te_loader_x_1):
                        # x_1 is coarse field if backward direction, fine field if forward direction
                        # x_0 is fine field if backward direction, coarse field if forward direction
                        te_x_0, te_x_1 = te_x_0.to(device), te_x_1.to(device)
                        te_s_path[0] = te_x_0
                        drift_norm.append(
                            euler_discretization(te_s_path, te_p_path, sample_nn, sigma)
                        )

                        # Compute L2 loss between generated field and target
                        generated_field = te_s_path[-1]

                        if direction == "fwd":
                            # if forward, we want to coarsen the generated fine field
                            # to calculate the L2 loss with the starting coarse field
                            # to ensure the coarsen
                            # field is of same regularity as the starting field
                            target_field = te_x_0
                            generated_field = coarsen_field(
                                generated_field, downsample_factor=1
                            )

                            l2_loss = th.nn.functional.mse_loss(
                                generated_field, target_field
                            )
                        else:
                            # if backward, we want to compare the generated coarse field
                            # with the target coarse field
                            target_field = te_x_1
                            l2_loss = th.nn.functional.mse_loss(
                                generated_field, target_field
                            )

                        l2_losses.append(l2_loss.item())

                    drift_norm = th.mean(th.cat(drift_norm)).item()
                    mean_l2_loss = np.mean(l2_losses)

                    denorm_fine = DatasetDenormalization(
                        fine_normalizer.mean, fine_normalizer.std
                    )
                    denorm_coarse = DatasetDenormalization(
                        coarse_normalizer.mean, coarse_normalizer.std
                    )

                    if rank == 0:
                        wandb.log(
                            {f"{direction}/test/drift_norm": drift_norm}, step=step
                        )

                    if rank == 0:
                        wandb.log(
                            {f"{direction}/test/l2_loss": mean_l2_loss}, step=step
                        )
                    if rank == 0:
                        console.log(f"mean L2 loss: {mean_l2_loss}")
                    for i, ti in enumerate(
                        resample_indices(discretization_steps + 1, 5)
                    ):
                        if rank == 0:
                            denorm_field = (
                                denorm_fine(te_s_path[ti])
                                if direction == "fwd"
                                else denorm_coarse(te_s_path[ti])
                            )
                            wandb.log(
                                {
                                    f"{direction}/test/x[{i}-{5}]": image_grid(
                                        denorm_field
                                    )
                                },
                                step=step,
                            )
                    for i, ti in enumerate(resample_indices(discretization_steps, 5)):
                        if rank == 0:
                            denorm_field = (
                                denorm_fine(te_p_path[ti])
                                if direction == "fwd"
                                else denorm_coarse(te_p_path[ti])
                            )
                            wandb.log(
                                {
                                    f"{direction}/test/p[{i}-{5}]": image_grid(
                                        denorm_field
                                    )
                                },
                                step=step,
                            )

            if step % loss_log_steps == 0:
                if rank == 0:
                    wandb.log(
                        {
                            f"{direction}/train/loss": loss.item(),
                            f"{direction}/train/base_loss": losses.mean().item(),
                            f"{direction}/train/operator_loss": (
                                losses_intOperator.mean().item()
                                if direction == "bwd"
                                else 0
                            ),
                            f"{direction}/train/intOp_scale_factor": intOp_scale_factor,
                        },
                        step=step,
                    )
                if rank == 0:
                    wandb.log({f"{direction}/train/grad_norm": grad_norm}, step=step)

            if step % imge_log_steps == 0:
                if rank == 0:
                    # Choose normalizer based on direction
                    denorm_0 = DatasetDenormalization(
                        (
                            coarse_normalizer.mean
                            if direction == "fwd"
                            else fine_normalizer.mean
                        ),
                        (
                            coarse_normalizer.std
                            if direction == "fwd"
                            else fine_normalizer.std
                        ),
                    )
                    denorm_1 = DatasetDenormalization(
                        (
                            fine_normalizer.mean
                            if direction == "fwd"
                            else coarse_normalizer.mean
                        ),
                        (
                            fine_normalizer.std
                            if direction == "fwd"
                            else coarse_normalizer.std
                        ),
                    )

                    # Denormalize before visualization
                    x_0_denorm = denorm_0(x_0)
                    x_1_denorm = denorm_1(x_1)

                    wandb.log(
                        {f"{direction}/train/x_0": image_grid(x_0_denorm, False)},
                        step=step,
                    )
                    wandb.log(
                        {f"{direction}/train/x_1": image_grid(x_1_denorm, False)},
                        step=step,
                    )

            if step % training_steps == 0:
                if rank == 0:
                    console.log(f"EMA update: {step}")
                # Make sure EMA is updated at the end of each iteration:
                ema.ema(sample_nn)

    progress.stop()


def save_checkpoint(saves, iteration, console):
    # Simplified from https://github.com/ghliu/SB-FBSDE/blob/main/util.py:
    checkpoint = {}
    i = 0
    fn = "./checkpoint/grf_10k_256_intOp_scaled.npz"
    keys = ["bwd_nn", "bwd_ema", "bwd_optim", "fwd_nn", "fwd_ema", "fwd_optim"]

    with th.cuda.device(rank):
        for k in keys:
            checkpoint[k] = saves[i].state_dict()
            i += 1
        checkpoint["iteration"] = iteration
        th.save(checkpoint, fn)
    if rank == 0:
        console.log(f"checkpoint saved: {fn}")


def restore_checkpoint(saves, console):
    # Simplified from https://github.com/ghliu/SB-FBSDE/blob/main/util.py:
    i = 0
    load_name = "./checkpoint/grf_10k_256_intOp_scaled.npz"
    assert load_name is not None
    if rank == 0:
        console.log(f"#loading checkpoint {load_name}...")

    with th.cuda.device(rank):
        checkpoint = th.load(load_name, map_location=th.device("cuda:%d" % rank))
        ckpt_keys = [*checkpoint.keys()][:-1]
        for k in ckpt_keys:
            saves[i].load_state_dict(checkpoint[k])
            i += 1
    if rank == 0:
        console.log("#successfully loaded all the modules")

    return checkpoint["iteration"]


if __name__ == "__main__":
    config = {
        "batch_dim": 64,  # Increased for better GPU utilization
        "cache_batch_dim": 2560,  #
        "cache_steps": 250,
        "test_steps": 5000,  # More frequent evaluation
        "iterations": 30,
        "training_steps": 5000,
        "load": False,  # continue training
    }

    fire.Fire(run(**config))
