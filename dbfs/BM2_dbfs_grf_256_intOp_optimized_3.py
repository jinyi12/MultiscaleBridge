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
import time

import fire
import numpy as np
import torch as th
import torch.profiler
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
from .models.transformer import OperatorTransformer
from .dct import dct_2d, idct_2d

# DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .models.bm2_transformer import BM2Transformer
from einops import repeat, rearrange

warnings.filterwarnings("ignore")

# EMA
from ema_pytorch import EMA
from .utils import make_data_grid

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the data file relative to the script location
coarse_data_path = os.path.join(BASE_DIR, "../Data/coarse_grf_10k.npy")
fine_data_path = os.path.join(BASE_DIR, "../Data/fine_grf_10k.npy")

# DDP
if th.cuda.is_available() and th.cuda.device_count() > 1:
    print("Initializing DDP")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = th.device(f"cuda:{rank}") if th.cuda.is_available() else th.device("cpu")
    parallel = True
else:
    print("Not initializing DDP")
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


def euler_discretization(
    x,
    nn,
    energy,
    chunk_size=128,
    store_control=False,
    direction_fn="fwd",
):
    """
    Performs an Euler–Maruyama discretization of the SDE controlling the process.

    Args:
        x (Tensor): Tensor of shape [T+1, cache_batch_dim, C, H, W] representing the state trajectory.
        nn (callable): The neural network drift function.
        energy (float): A scalar value used in computing the diffusion factor.
        chunk_size (int, optional): Size of the sub-batches; default is 128.
        store_control (bool, optional): If True, store the intermediate control values.
        direction_fn (str, optional): The function to use for the direction of the discretization.
                                        If "fwd", integrates using forward drift.
                                        If "bwd", integrates using backward drift.

    Returns:
        drift_norms (Tensor): Average squared control norm over the discretization.
        xp (Tensor or None): If store_control is True, an array storing intermediate control updates;
                             otherwise, None.
    """
    # x has shape [T+1, cache_batch_dim, C, H, W]
    T = x.shape[0] - 1  # number of discretization steps
    B = x.shape[1]  # B = cache_batch_dim

    dt_val = 1.0 / T
    t_start = 0.0
    t_end = 1.0
    dt = th.full((B,), dt_val, device=x.device)

    drift_norms = 0.0
    _, b, c, h, w = x.shape

    # make data grid for batch dimension relative operations
    data_grid = make_data_grid(res=h)
    data_grid = repeat(data_grid, "b hw c -> (repeat b) hw c", repeat=chunk_size)
    data_grid = data_grid.to(x.device)

    freqs = np.pi * th.linspace(0, h - 1, h) / h
    freq = (freqs[:, None] ** 2 + freqs[None, :] ** 2).to(x.device)
    frequencies_squared = freq + 1.0
    a_k = frequencies_squared[None, None]
    sigma_k = th.pow(a_k, -0.01) / energy

    # Initialize xp if store_control is True
    if store_control:
        xp = th.zeros_like(x[:-1])
    else:
        xp = None

    for i in range(1, T + 1):
        # Current time based on direction
        t_val = (i - 1) * dt_val
        t_tensor = th.full((B,), t_val, device=x.device)

        # Process the neural network in chunks
        alpha_chunks = []

        # Split the batch dimension into sub-batches of size chunk_size
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk_input = x[i - 1][start:end]
            chunk_t = t_tensor[start:end]

            # nn returns both forward and backward drifts
            fwd_drift, bwd_drift = nn(chunk_input, chunk_t, input_pos=data_grid)
            # Select the appropriate drift
            alpha_chunk = fwd_drift if direction_fn == "fwd" else bwd_drift

            alpha_chunks.append(alpha_chunk)

        # Concatenate results
        alpha_t = th.cat(alpha_chunks, dim=0)
        drift_norms = drift_norms + th.mean(alpha_t.view(B, -1) ** 2, dim=1)

        # Convert to spectral domain using DCT
        alpha_t_spec = dct_2d(alpha_t, norm="ortho")
        x_i_spec = dct_2d(x[i - 1], norm="ortho")

        # Compute time tensors for control computation
        t_ = t_tensor[:, None, None, None]
        t_end_tensor = th.ones_like(t_)  # spectral euler only works for t_end = 1

        # Compute control in spectral domain
        control = (
            a_(t_, t_end_tensor, a_k) * alpha_t_spec
            - a_2(t_, t_end_tensor, a_k) * x_i_spec
        ) / v_(t_, t_end_tensor, a_k)
        if store_control:
            xp[i - 1] = idct_2d(control, norm="ortho")

        # Calculate drift in spectral domain
        drift_t = (-a_k * x_i_spec + control) * dt[:, None, None, None]

        # Add random noise with proper scaling for diffusion
        eps_t = dct_2d(th.randn_like(x[i - 1]), norm="ortho")

        # No diffusion on the last step
        if i == T:
            diffusion_t = 0
        else:
            # Use abs(dt) for the square root to avoid numerical issues
            diffusion_t = sigma_k * th.sqrt(th.abs(dt)[:, None, None, None]) * eps_t

        # Update the state for the next time step
        x[i] = idct_2d(x_i_spec + drift_t + diffusion_t, norm="ortho")

    drift_norms = drift_norms / T
    return drift_norms.cpu(), xp


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
        # Use the provided train_data_path; here it should be the absolute path.
        train_data = np.load(train_data_path)
        self.mean = train_data.mean()
        self.std = train_data.std()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = th.from_numpy(x).float()
        return (x - self.mean) / self.std


# Add denormalization class
class DatasetDenormalization:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x * self.std + self.mean


# Now, when creating the DatasetNormalization instance,
# use the absolute path computed above.
coarse_normalizer = DatasetNormalization(coarse_data_path)
fine_normalizer = DatasetNormalization(fine_data_path)

# Normalization transform
transform_coarse = transforms.Compose([ToTensorCustom(), coarse_normalizer])

transform_fine = transforms.Compose([ToTensorCustom(), fine_normalizer])

# Load datasets for coarse (source) and fine (target) fields
tr_data_0 = FieldsDataset(
    data_path=coarse_data_path,
    train=True,
    transform=transform_coarse,
)
te_data_0 = FieldsDataset(
    data_path=coarse_data_path,
    train=False,
    transform=transform_coarse,
)
tr_data_1 = FieldsDataset(
    data_path=fine_data_path,
    train=True,
    transform=transform_fine,
)
te_data_1 = FieldsDataset(
    data_path=fine_data_path,
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
        in_channel=1,
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
    in_axis=2,
    out_axis=2,
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
    two_stage_training=True,  # Enable two-stage training as described in the paper
    consistency_loss_weight=1.0,  # Weight for the consistency loss term
    t_epsilon=1e-5,  # Avoid singularities by sampling t from (epsilon, 1-epsilon)
    use_checkpointing=True,
    run_name=None,
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

    # Initialize wandb with a different project (only on rank 0 if using DDP)
    if rank == 0:
        wandb.init(project="BM2_GRF_MultiscaleBridge", config=config)
    # Setup data:
    tr_iter_0 = train_iter(tr_data_0, batch_dim, tr_spl_0)
    tr_iter_1 = train_iter(tr_data_1, batch_dim, tr_spl_1)
    tr_cache_iter_0 = train_iter(tr_data_0, cache_batch_dim, tr_spl_0)
    tr_cache_iter_1 = train_iter(tr_data_1, cache_batch_dim, tr_spl_1)
    te_loader_0 = test_loader(te_data_0, test_batch_dim, te_spl_0)
    te_loader_1 = test_loader(te_data_1, test_batch_dim, te_spl_1)

    # Precompute discretization times:
    t_T = 1.0
    ts = th.linspace(0, t_T, discretization_steps + 1, device=device)
    ts_idx = th.round(ts * discretization_steps).long()
    ts_idx[-1] = discretization_steps

    # data dimension info
    b, c, h, w = next(tr_iter_0).shape

    # # make a grid for the input and output positions
    # # for use with batch dimension relative operations
    # data_grid = make_data_grid(res=h)
    # data_grid = repeat(data_grid, "b hw c -> (repeat b) hw c", repeat=batch_dim)
    # data_grid = data_grid.to(device)

    # # make a grid for the test positions
    # test_data_grid = make_data_grid(res=64)
    # test_data_grid = repeat(
    #     test_data_grid,
    #     "b hw c -> (repeat b) hw c",
    #     repeat=test_batch_dim,
    # )
    # test_data_grid = test_data_grid.to(device)

    # For BM², we use a single neural network for both forward and backward drift
    nn = BM2Transformer(
        in_axis=in_axis,
        out_axis=out_axis,
        in_channel=1,
        out_channel=1,
        pos_dim=256,
        latent_dim=256,
        num_heads=4,
        depth_enc=6,
        depth_dec=2,
        scale=1,
        self_per_cross_attn=1,
        height=32,
        use_checkpointing=use_checkpointing,
    ).to(device)

    nn = torch.compile(nn)
    optim = th.optim.Adam(nn.parameters(), lr=learning_rate)

    # EMA helper for the model parameters
    ema = EMA(nn, beta=ema_decay, update_after_step=0, update_every=1, power=3 / 4)

    # # Sample model used for generating paths (with EMA parameters)
    # sample_nn = ema.ema_copy()

    step = 0
    start_iteration = 0
    if load:
        # Load checkpoint if available
        start_iteration, step = restore_checkpoint(
            # {"nn": nn, "optim": optim, "ema": ema, "sample_nn": sample_nn},
            {
                "nn": nn,
                "optim": optim,
                "ema": ema,
            },  # dont need sample_nn, as we can use `ema` directly for sampling
            console,
            run_name=wandb.run.name,
        )
        if rank == 0:
            console.log(f"Loaded checkpoint: iteration {start_iteration}, step {step}")

    step_t = progress.add_task("step", total=iterations * training_steps)

    scaler = th.amp.GradScaler(enabled=True)

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
            # fwd_target_t = fwd_target(x_t_forward, x_1, t)
            fwd_target_t = fwd_target(x_t_forward, x_1, t=None)  # use DBFS formulation

            # Backward pass
            x_t_backward = sample_bridge(x_0, x_1, t, sigma)
            # bwd_target_t = bwd_target(x_t_backward, x_0, t)
            bwd_target_t = bwd_target(x_t_backward, x_0, t=None)  # use DBFS formulation

            with th.autocast(device_type="cuda", dtype=th.float16, enabled=True):
                with torch.profiler.record_function("BM2_Stage1_ForwardBackward"):
                    # Predict both forward and backward drifts with the same network
                    # The network can distinguish between forward and backward based on a direction flag
                    fwd_pred, _ = nn(x_t_forward, t, input_pos=None)
                    _, bwd_pred = nn(x_t_backward, t, input_pos=None)

                    # Independent BM loss
                    fwd_losses = (fwd_target_t - fwd_pred) ** 2
                    bwd_losses = (bwd_target_t - bwd_pred) ** 2

                    fwd_losses = th.mean(
                        fwd_losses.reshape(fwd_losses.shape[0], -1), dim=1
                    )
                    bwd_losses = th.mean(
                        bwd_losses.reshape(bwd_losses.shape[0], -1), dim=1
                    )

                    # Add integral operator loss for forward direction
                    if intOp_scale_factor > 0:
                        # For forward direction: x_1 is the fine field, and bwd_target_t contains coarse field information
                        losses_intOperator = (
                            bwd_target_t - coarsen_field(x_1, downsample_factor=1)
                        ) ** 2
                        losses_intOperator = th.mean(
                            losses_intOperator.reshape(losses_intOperator.shape[0], -1),
                            dim=1,
                        )

                        # Add weighted integral operator loss to forward loss
                        fwd_losses = (
                            fwd_losses + intOp_scale_factor * losses_intOperator
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

    # Initialize cache for endpoints
    fwd_cache_x0 = []
    fwd_cache_x1 = []
    bwd_cache_x0 = []
    bwd_cache_x1 = []

    # Define function to sample the coupling
    def sample_dbfs_coupling(step):
        nonlocal fwd_cache_x0, fwd_cache_x1, bwd_cache_x0, bwd_cache_x1

        if iteration == 1:
            # Independent coupling for first iteration:
            f_0 = next(tr_iter_0).to(device)
            f_1 = next(tr_iter_1).to(device)
            b_1 = f_1
            b_0 = f_0
        else:
            with th.inference_mode():
                # Check if it's time to update cache
                # print("step -1 % cache_steps", (step - 1) % cache_steps)
                # print("step", step)
                if (step - 1) % cache_steps == 0 and step != 1:
                    if rank == 0:
                        console.log(f"cache update: {step}")

                    # Update sample_nn with latest EMA parameters
                    # ema.ema(sample_nn) # dont need to do this, as we can use `ema` directly for sampling

                    # Update forward path endpoints
                    f_0_cache_batch = next(tr_cache_iter_0).to(device)

                    # Update backward path endpoint
                    b_1_cache_batch = next(tr_cache_iter_1).to(device)

                    # Compute chunk_size as the minimum between the number of available samples in x_0 and cache_batch_dim.
                    # This adaptive computation ensures that when the cache is nearly exhausted, we only process the remaining samples.
                    chunk_size = int(min(f_0_cache_batch.size(0), cache_batch_dim / 4))

                    print("Chunk size", chunk_size)

                    # Initialize trajectory array
                    fwd_traj = th.zeros(
                        (
                            discretization_steps + 1,
                            f_0_cache_batch.shape[0],
                            *f_0_cache_batch.shape[1:],
                        ),
                        device=device,
                    )
                    fwd_traj[0] = f_0_cache_batch  # Set initial condition

                    # Initialize trajectory array
                    bwd_traj = th.zeros(
                        (
                            discretization_steps + 1,
                            b_1_cache_batch.shape[0],
                            *b_1_cache_batch.shape[1:],
                        ),
                        device=device,
                    )
                    bwd_traj[0] = b_1_cache_batch  # Set initial condition

                    # Use euler_discretization to simulate backward process
                    drift_norms, _ = euler_discretization(
                        x=fwd_traj,
                        # nn=sample_nn,
                        nn=ema,
                        energy=sigma,
                        chunk_size=chunk_size,
                        store_control=False,
                        direction_fn="fwd",
                    )
                    # backward iteration, we simulate the forward, and then reverse the path
                    bwd_cache_x0 = fwd_traj[-1].detach()
                    bwd_cache_x1 = fwd_traj[0]

                    # Use euler_discretization to simulate forward process
                    drift_norms, _ = euler_discretization(
                        x=bwd_traj,
                        # nn=sample_nn,
                        nn=ema,
                        energy=sigma,
                        chunk_size=chunk_size,
                        store_control=False,
                        direction_fn="bwd",
                    )

                    fwd_cache_x0 = bwd_traj[-1].detach()
                    fwd_cache_x1 = bwd_traj[0]

                # Randomly select samples from cache
                idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
                # bwd_idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]

                # Get forward path endpoints
                f_0 = fwd_cache_x0[idx]
                f_1 = fwd_cache_x1[idx]

                # Get backward path endpoints
                b_1 = bwd_cache_x0[
                    idx
                ]  # consistent naming with the paper, b_1 is the fine field
                b_0 = bwd_cache_x1[
                    idx
                ]  # consistent naming with the paper, b_0 is the coarse field

        return f_0, f_1, b_1, b_0

    for iteration in range(start_iteration + 1, iterations + 1):
        if rank == 0:
            console.log(f"BM² iteration {iteration}: {step}")
        if rank == 0:
            progress.update(iteration_t, completed=iteration)

        # BM² training loop
        for step in range(step + 1, step + training_steps + 1):
            progress.update(step_t, completed=step)
            optim.zero_grad()

            # Get the coupled samples
            # f_1 from euler discretization is the fine field
            # b_0 from euler discretization is the coarse field
            # forward direction: reverse path -> use backward drift -> x_1 = b_1, x_0 = b_0
            # backward direction: forward path -> use forward drift -> x_0 = f_0, x_1 = f_1
            f_0, f_1, b_1, b_0 = sample_dbfs_coupling(step)

            # Sample t avoiding singularities
            t = (
                th.rand(size=(batch_dim,), device=device) * (1.0 - 2 * t_epsilon)
                + t_epsilon
            )

            pi_f_t = sample_bridge(x_0=f_0, x_1=f_1, t=t, energy=sigma)
            pi_b_t = sample_bridge(x_0=b_1, x_1=b_0, t=t, energy=sigma)

            # Compute targets and model predictions
            with th.autocast(device_type="cuda", dtype=th.float16, enabled=True):
                with torch.profiler.record_function("BM2_Training_ForwardBackward"):
                    # # Forward targets and predictions
                    target_f_t = fwd_target(x_t=pi_b_t, x_1=f_1, t=None)
                    prediction_f_t, _ = nn(pi_f_t, t, input_pos=None)

                    # Backward targets and predictions
                    target_b_t = bwd_target(x_t=pi_f_t, x_0=b_0, t=None)
                    _, prediction_b_t = nn(pi_b_t, t, input_pos=None)

                    # Compute the BM² coupled loss
                    loss_f_t = (target_f_t - prediction_f_t) ** 2
                    loss_b_t = (target_b_t - prediction_b_t) ** 2

                    loss_f_t = th.mean(loss_f_t.reshape(loss_f_t.shape[0], -1), dim=1)
                    loss_b_t = th.mean(loss_b_t.reshape(loss_b_t.shape[0], -1), dim=1)

                    # Add integral operator loss for backward path
                    # This enforces that the backward drift respects the coarse-graining operation
                    if intOp_scale_factor > 0:
                        # f_1 is the fine field (generated by the forward model)
                        # and target_b_t is the coarse field f_0.
                        losses_intOperator = (
                            target_b_t - coarsen_field(b_1, downsample_factor=1)
                        ) ** 2

                        # print("losses_intOperator", losses_intOperator)

                        losses_intOperator = th.mean(
                            losses_intOperator.reshape(losses_intOperator.shape[0], -1),
                            dim=1,
                        )

                        # Add weighted integral operator loss to forward loss
                        loss_f_t = loss_f_t + intOp_scale_factor * losses_intOperator

                    loss = th.mean(loss_f_t + loss_b_t)

            # Backpropagation and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            grad_norm = th.nn.utils.clip_grad_norm_(nn.parameters(), grad_max_norm)
            scaler.step(optim)
            scaler.update()
            ema.update()

            # Logging
            if step % loss_log_steps == 0 and rank == 0:
                console.log(f"BM² Step {step}: loss {loss:.4f}, grad {grad_norm:.4f}")

                # Log integral operator loss if it's being used
                if intOp_scale_factor > 0:
                    wandb.log(
                        {
                            "bm2/intOp_loss": intOp_scale_factor
                            * th.mean(losses_intOperator).item(),
                            "bm2/total_fwd_loss": th.mean(loss_f_t).item(),
                            "bm2/total_bwd_loss": th.mean(loss_b_t).item(),
                            "bm2/total_loss": loss.item(),
                        },
                        step=step,
                    )

            # Checkpointing
            if step % imge_log_steps == 0 and rank == 0:
                denorm_fine = DatasetDenormalization(
                    fine_normalizer.mean, fine_normalizer.std
                )
                denorm_coarse = DatasetDenormalization(
                    coarse_normalizer.mean, coarse_normalizer.std
                )
                fwd_x_0 = denorm_coarse(f_0)
                fwd_x_1 = denorm_fine(f_1)
                bwd_x_0 = denorm_fine(b_1)
                bwd_x_1 = denorm_coarse(b_0)
                wandb.log(
                    {
                        "bm2/train/fwd_x_0": wandb.Image(fwd_x_0),
                        "bm2/train/fwd_x_1": wandb.Image(fwd_x_1),
                        "bm2/train/bwd_x_0": wandb.Image(bwd_x_0),
                        "bm2/train/bwd_x_1": wandb.Image(bwd_x_1),
                    },
                    step=step,
                )

            # Evaluation using current model
            if step % (test_steps) == 0 and rank == 0:
                # Update sample_nn with latest EMA parameters
                # ema.ema(sample_nn)

                saves = [nn, ema, optim]
                run_name = wandb.run.name
                save_checkpoint(saves, step, console, run_name)

                # Determine the checkpoint file path (as used in save_checkpoint)
                checkpoint_file = os.path.join(
                    BASE_DIR,
                    f"./checkpoint/{run_name}_BM2_dbfs_grf_10k_256_intOp_scaled_2.npz",
                )

                # Create a wandb Artifact for the model checkpoint.
                # Using a consistent artifact name will allow wandb to version control it.
                model_artifact = wandb.Artifact(
                    f"{run_name}_model_checkpoint",
                    type="model_checkpoint",
                    description=f"Model checkpoint for {run_name} at step {step}",
                )
                # Add the checkpoint file to the artifact.
                model_artifact.add_file(checkpoint_file)
                # Log the artifact (wandb will automatically version it).
                wandb.log_artifact(model_artifact)

                # Evaluate on test set
                with th.no_grad():
                    test_loss_fwd = []
                    test_loss_bwd = []
                    test_loss_intOperator = []
                    num_batches = 0

                    for f_0_test, b_1_test in zip(te_loader_0, te_loader_1):
                        f_0_test = f_0_test.to(device)
                        b_1_test = b_1_test.to(device)
                        t_test = (
                            th.rand(size=(f_0_test.shape[0],), device=device)
                            * (1.0 - 2 * t_epsilon)
                            + t_epsilon
                        )  # t in [epsilon, 1-epsilon]

                        # create forward endpoints
                        fwd_traj = th.zeros(
                            (
                                discretization_steps + 1,
                                f_0_test.shape[0],
                                1,
                                32,
                                32,
                            ),
                            device=device,
                        )
                        fwd_traj[0] = f_0_test

                        # create backward endpoints
                        bwd_traj = th.zeros(
                            (
                                discretization_steps + 1,
                                b_1_test.shape[0],
                                1,
                                32,
                                32,
                            ),
                            device=device,
                        )
                        bwd_traj[0] = b_1_test

                        # simulate forward process
                        _, _ = euler_discretization(
                            x=fwd_traj,
                            # nn=sample_nn,
                            nn=ema,
                            energy=sigma,
                            chunk_size=f_0_test.shape[0],
                            store_control=False,
                            direction_fn="fwd",
                        )
                        f_1_test = fwd_traj[-1]

                        # simulate backward process
                        _, _ = euler_discretization(
                            x=bwd_traj,
                            # nn=sample_nn,
                            nn=ema,
                            energy=sigma,
                            chunk_size=b_1_test.shape[0],
                            store_control=False,
                            direction_fn="bwd",
                        )
                        b_0_test = bwd_traj[-1]

                        # Denormalize using the corresponding dataset denormalizers
                        denorm_fine = DatasetDenormalization(
                            fine_normalizer.mean, fine_normalizer.std
                        )
                        denorm_coarse = DatasetDenormalization(
                            coarse_normalizer.mean, coarse_normalizer.std
                        )

                        # Apply denormalization to the outputs/targets
                        f_1_test_denorm = denorm_fine(f_1_test)
                        b_1_test_denorm = denorm_fine(b_1_test)
                        b_0_test_denorm = denorm_coarse(b_0_test)
                        f_0_test_denorm = denorm_coarse(f_0_test)

                        # Then compute relative errors in physical (denormalized) units:
                        rel_err_f_test = th.norm(
                            (f_1_test_denorm - b_1_test_denorm).reshape(
                                f_1_test_denorm.shape[0], -1
                            ),
                            p=2,
                            dim=1,
                        ) / th.norm(
                            b_1_test_denorm.reshape(b_1_test_denorm.shape[0], -1),
                            p=2,
                            dim=1,
                        )

                        rel_err_b_test = th.norm(
                            (b_0_test_denorm - f_0_test_denorm).reshape(
                                b_0_test_denorm.shape[0], -1
                            ),
                            p=2,
                            dim=1,
                        ) / th.norm(
                            f_0_test_denorm.reshape(f_0_test_denorm.shape[0], -1),
                            p=2,
                            dim=1,
                        )

                        if intOp_scale_factor > 0:
                            # For the integral operator, compare the coarsened f_1_test with f_0_test:
                            rel_err_int = th.norm(
                                (
                                    coarsen_field(f_1_test_denorm, downsample_factor=1)
                                    - f_0_test_denorm
                                ).reshape(f_0_test_denorm.shape[0], -1),
                                p=2,
                                dim=1,
                            ) / th.norm(
                                f_0_test_denorm.reshape(f_0_test_denorm.shape[0], -1),
                                p=2,
                                dim=1,
                            )
                            rel_err_int = rel_err_int.mean()
                        else:
                            rel_err_int = torch.tensor(0.0, device=device)

                        test_loss_fwd.append(rel_err_f_test)
                        test_loss_bwd.append(rel_err_b_test)
                        test_loss_intOperator.append(rel_err_int)

                    test_loss_fwd = th.stack(test_loss_fwd)
                    test_loss_bwd = th.stack(test_loss_bwd)
                    test_loss_intOperator = th.stack(test_loss_intOperator)

                    test_loss_fwd = test_loss_fwd.mean()
                    test_loss_bwd = test_loss_bwd.mean()
                    test_loss_intOperator = test_loss_intOperator.mean()

                    if rank == 0:
                        console.log(
                            f"Test mean relative L2 error forward: {test_loss_fwd:.6f}"
                        )
                        console.log(
                            f"Test mean relative L2 error backward: {test_loss_bwd:.6f}"
                        )
                        console.log(
                            f"Test mean relative L2 error intOperator: {intOp_scale_factor * test_loss_intOperator:.6f}"
                        )

                        wandb.log(
                            {
                                "bm2/test/mean_rel_err_fwd": test_loss_fwd,
                                "bm2/test/mean_rel_err_bwd": test_loss_bwd,
                                "bm2/test/mean_rel_err_intOp": intOp_scale_factor
                                * test_loss_intOperator,
                                "bm2/test/mean_rel_err_total": test_loss_fwd
                                + test_loss_bwd
                                + intOp_scale_factor * test_loss_intOperator,
                            },
                            step=step,
                        )

                        # ########################### High Resolution ###########################
                        # Forward direction high-res testing
                        console.log("High Resolution Testing")
                        with th.no_grad():
                            te_s_path = th.zeros(
                                size=(discretization_steps + 1,)
                                + (test_batch_dim, 1, 64, 64),
                                device=device,
                            )
                            te_p_path = th.zeros(
                                size=(discretization_steps,)
                                + (test_batch_dim, 1, 64, 64),
                                device=device,
                            )
                            # Physical field, dont assume range of [0, 1]
                            drift_norm = []

                            denorm_fine = DatasetDenormalization(
                                fine_normalizer.mean, fine_normalizer.std
                            )
                            denorm_coarse = DatasetDenormalization(
                                coarse_normalizer.mean, coarse_normalizer.std
                            )

                            # Forward direction high-res testing
                            for te_x_0, te_x_1 in zip(te_loader_0, te_loader_1):
                                te_x_0 = F.resize(te_x_0, 64).to(device)
                                te_x_1 = F.resize(te_x_1, 64).to(device)

                                te_s_path[0] = te_x_0
                                drift_norm, te_p_path = euler_discretization(
                                    x=te_s_path,
                                    nn=ema,
                                    energy=sigma,
                                    chunk_size=te_x_0.shape[0],
                                    store_control=True,
                                    direction_fn="fwd",
                                )
                                break

                            # Log drift norm
                            if rank == 0:
                                wandb.log(
                                    {"bm2/test_high_res/fwd/drift_norm": drift_norm},
                                    step=step,
                                )

                            # Log high-resolution forward test images
                            for i, ti in enumerate(
                                resample_indices(discretization_steps + 1, 5)
                            ):
                                if rank == 0:
                                    denorm_field = denorm_fine(te_s_path[ti])
                                    wandb.log(
                                        {
                                            f"bm2/test_high_res/fwd/x[{i}-{5}]": image_grid(
                                                denorm_field
                                            )
                                        },
                                        step=step,
                                    )
                            for i, ti in enumerate(
                                resample_indices(discretization_steps, 5)
                            ):
                                if rank == 0:
                                    denorm_field = denorm_fine(te_p_path[ti])
                                    wandb.log(
                                        {
                                            f"bm2/test_high_res/fwd/p[{i}-{5}]": image_grid(
                                                denorm_field
                                            )
                                        },
                                        step=step,
                                    )

                            # Backward direction high-res testing
                            te_s_path = th.zeros(
                                size=(discretization_steps + 1,)
                                + (test_batch_dim, 1, 64, 64),
                                device=device,
                            )
                            te_p_path = th.zeros(
                                size=(discretization_steps,)
                                + (test_batch_dim, 1, 64, 64),
                                device=device,
                            )
                            drift_norm = []

                            te_s_path[0] = te_x_1
                            drift_norm, te_p_path = euler_discretization(
                                x=te_s_path,
                                nn=ema,
                                energy=sigma,
                                chunk_size=te_x_1.shape[0],
                                store_control=True,
                                direction_fn="bwd",
                            )

                            # Log drift norm
                            if rank == 0:
                                wandb.log(
                                    {"bm2/test_high_res/bwd/drift_norm": drift_norm},
                                    step=step,
                                )

                            # Log high-resolution backward test images
                            for i, ti in enumerate(
                                resample_indices(discretization_steps + 1, 5)
                            ):
                                if rank == 0:
                                    denorm_field = denorm_coarse(te_s_path[ti])
                                    wandb.log(
                                        {
                                            f"bm2/test_high_res/bwd/x[{i}-{5}]": image_grid(
                                                denorm_field
                                            )
                                        },
                                        step=step,
                                    )
                            for i, ti in enumerate(
                                resample_indices(discretization_steps, 5)
                            ):
                                if rank == 0:
                                    denorm_field = denorm_coarse(te_p_path[ti])
                                    wandb.log(
                                        {
                                            f"bm2/test_high_res/bwd/p[{i}-{5}]": image_grid(
                                                denorm_field
                                            )
                                        },
                                        step=step,
                                    )
            if step % training_steps == 0:
                if rank == 0:
                    console.log(f"Updating EMA weights at step: {step}")
                ema.update_model_with_ema()  # Switch EMA implementation, where the model `nn` gets updated with the EMA-averaged weights
                # Now our unified network `nn` holds the EMA-averaged weights

    progress.stop()

    return nn


def save_checkpoint(saves, iteration, console, run_name):
    # Simplified from https://github.com/ghliu/SB-FBSDE/blob/main/util.py:
    checkpoint = {}
    i = 0
    # Include run name in the file name:
    fn = os.path.join(
        BASE_DIR, f"./checkpoint/{run_name}_BM2_dbfs_grf_10k_256_intOp_scaled_2.npz"
    )
    keys = ["nn", "ema", "optim"]

    with th.cuda.device(rank):
        for k in keys:
            checkpoint[k] = saves[i].state_dict()
            i += 1
        checkpoint["iteration"] = iteration
        th.save(checkpoint, fn)
    if rank == 0:
        console.log(f"checkpoint saved: {fn}")


def restore_checkpoint(saves, console, run_name):
    # Simplified from https://github.com/ghliu/SB-FBSDE/blob/main/util.py:
    # Use the run name in the load path:
    load_name = os.path.join(
        BASE_DIR, f"./checkpoint/{run_name}_BM2_dbfs_grf_10k_256_intOp_scaled_2.npz"
    )
    assert load_name is not None
    if rank == 0:
        console.log(f"#loading checkpoint {load_name}...")

    with th.cuda.device(rank):
        checkpoint = th.load(load_name, map_location=th.device("cuda:%d" % rank))
        ckpt_keys = [*checkpoint.keys()][:-1]
        for k in ckpt_keys:
            print("k", k)
            saves[k].load_state_dict(checkpoint[k])
    if rank == 0:
        console.log("#successfully loaded all the modules")

    return checkpoint["iteration"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train BM2_FS model with integral operator scaling and checkpointing"
    )
    parser.add_argument(
        "--in_axis", type=int, default=2, help="Input axis (default: 2)"
    )
    parser.add_argument(
        "--out_axis", type=int, default=2, help="Output axis (default: 2)"
    )
    parser.add_argument(
        "--batch_dim",
        type=int,
        default=128,
        help="Batch dimension (increased for better GPU utilization)",
    )
    parser.add_argument(
        "--cache_batch_dim", type=int, default=2560, help="Cache batch dimension"
    )
    parser.add_argument(
        "--cache_steps", type=int, default=250, help="Number of cache steps"
    )
    parser.add_argument(
        "--test_steps", type=int, default=5000, help="Number of test steps"
    )
    parser.add_argument(
        "--iterations", type=int, default=30, help="Number of iterations"
    )
    parser.add_argument(
        "--training_steps", type=int, default=5000, help="Number of training steps"
    )
    parser.add_argument(
        "--load", type=bool, default=False, help="Continue training from checkpoint"
    )
    parser.add_argument(
        "--intOp_scale_factor",
        type=float,
        default=1,
        help="Integral operator scale factor",
    )
    parser.add_argument(
        "--two_stage_training", action="store_true", help="Enable two-stage training"
    )
    parser.add_argument(
        "--use_checkpointing",
        action="store_true",
        default=True,
        help="Use checkpointing during training",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name to load checkpoint from if resuming training.",
    )
    args = parser.parse_args()

    if args.run_name is not None:
        # Use the provided run name for checkpoint naming and restoration.
        run_name = args.run_name
        print(f"Resuming from checkpoint of run: {run_name}")

    # Pass the run_name to functions that use it so that the proper checkpoint can be located.
    run(**vars(args))
