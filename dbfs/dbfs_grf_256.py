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

warnings.filterwarnings("ignore")

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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from models.transformer import OperatorTransformer
from dct import dct_2d, idct_2d

# DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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


def euler_discretization(x, xp, nn, energy):
    # Assumes x has shape [T, B, C, H, W].
    # Assumes x[0] already initialized.
    # We normalize by D = C * H * W the drift squared norm, and not by scalar sigma.
    # Fills x[1] to x[T] and xp[0] to xp[T - 1].
    T = x.shape[0] - 1  # Discretization steps.
    B = x.shape[1]
    dt = th.full(size=(x.shape[1],), fill_value=1.0 / T, device=device)
    drift_norms = 0.0

    _, b, c, h, w = x.shape

    freqs = np.pi * th.linspace(0, h - 1, h) / h
    freq = (freqs[:, None] ** 2 + freqs[None, :] ** 2).to(x.device)
    frequencies_squared = freq + 1.0
    a_k = frequencies_squared[None, None]
    sigma_k = th.pow(a_k, -0.01) / energy

    for i in range(1, T + 1):
        t = dt * (i - 1)
        alpha_t = nn(x[i - 1], t)
        drift_norms = drift_norms + th.mean(alpha_t.view(B, -1) ** 2, dim=1)
        alpha_t = dct_2d(alpha_t, norm="ortho")
        x_i = dct_2d(x[i - 1], norm="ortho")
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
        x[i] = idct_2d(x_i + drift_t + diffusion_t, norm="ortho")

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


# def init_nn():
#     return OperatorTransformer(
#         in_channel=2,
#         out_channel=1,
#         latent_dim=256,
#         pos_dim=256,
#         num_heads=4,
#         # depth_enc=6,
#         depth_enc=6,
#         depth_dec=2,
#         scale=1,
#         self_per_cross_attn=1,
#         height=32,  # to match the fine fields
#     ).to(device)
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
        height=32,  # to match the fine fields
    ).to(device)


# def init_nn():
#     return OperatorTransformer(
#         in_channel=2,
#         out_channel=1,
#         latent_dim=128,  # reduced latent dimension
#         pos_dim=128,  # smaller positional dimension
#         num_heads=4,  # fewer heads
#         depth_enc=4,  # fewer encoder layers
#         depth_dec=2,  # fewer decoder layers
#         scale=1,
#         self_per_cross_attn=1,
#         height=32,
#     ).to(device)


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


# def run(
#     method=DBFS,
#     sigma=1.0,
#     # iterations=60,
#     iterations=60,
#     training_steps=5000,
#     discretization_steps=30,
#     batch_dim=32,
#     learning_rate=1e-4,
#     grad_max_norm=1.0,
#     ema_decay=0.999,
#     cache_steps=250,
#     cache_batch_dim=2560,
#     # test_steps=5000,
#     test_steps=10,
#     # test_batch_dim=512,
#     test_batch_dim=32,
#     loss_log_steps=100,
#     imge_log_steps=1000,
#     load=False,
# ):
def run(
    method=DBFS,
    sigma=1.0,
    iterations=60,  # fewer iterations
    training_steps=5000,  # fewer training steps
    discretization_steps=30,
    batch_dim=32,  # smaller batch size
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
):
    ...
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
    step_t = progress.add_task("step", total=iterations * training_steps)

    if rank == 0:
        wandb.init(project="dbfs", config=config, mode="online")
    if rank == 0:
        console.log(wandb.config)

    # DDP
    if parallel:
        world_size = dist.get_world_size()
        if batch_dim % world_size != 0:
            raise ValueError(
                f"minibatch_size ({batch_dim}) must be divisible by world_size ({world_size})"
            )
        batch_dim = batch_dim // world_size
        if cache_batch_dim % world_size != 0:
            raise ValueError(
                f"cache_batch_dim ({cache_batch_dim}) must be divisible by world_size ({world_size})"
            )
        cache_batch_dim = cache_batch_dim // world_size
        if test_batch_dim % world_size != 0:
            raise ValueError(
                f"test_batch_dim ({test_batch_dim}) must be divisible by world_size ({world_size})"
            )
        test_batch_dim = test_batch_dim // world_size
        if rank == 0:
            console.log(f"minibatch_size per GPU: {batch_dim}")
            console.log(f"cache_batch_dim per GPU: {cache_batch_dim}")
            console.log(f"test_batch_dim per GPU: {test_batch_dim}")

    tr_iter_0 = train_iter(tr_data_0, batch_dim, tr_spl_0)
    tr_iter_1 = train_iter(tr_data_1, batch_dim, tr_spl_1)
    tr_cache_iter_0 = train_iter(tr_data_0, cache_batch_dim, tr_spl_0)
    tr_cache_iter_1 = train_iter(tr_data_1, cache_batch_dim, tr_spl_1)
    te_loader_0 = test_loader(te_data_0, test_batch_dim, te_spl_0)
    te_loader_1 = test_loader(te_data_1, test_batch_dim, te_spl_1)

    # console.log("Len of te_loader_0: ", len(te_loader_0))
    # console.log("Len of te_loader_1: ", len(te_loader_1))

    bwd_nn = init_nn().to(device)
    fwd_nn = init_nn().to(device)
    # DDP
    if parallel:
        bwd_nn = DDP(bwd_nn, device_ids=[device])
        fwd_nn = DDP(fwd_nn, device_ids=[device])

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
        size=(discretization_steps + 1,) + (cache_batch_dim, 1, 32, 32), device=device
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

                            euler_discretization(s_path, p_path, fwd_sample_nn, sigma)
                        # Random selection:
                        idx = th.randperm(cache_batch_dim, device=device)[:batch_dim]
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

            x_0, x_1 = sample_dbfs_coupling(step)
            t = th.rand(size=(batch_dim,), device=device) * t_T
            x_t = sample_bridge(x_0, x_1, t, sigma)
            target_t = dbfs_target(x_t, x_1, t)

            with th.autocast(device_type="cuda", dtype=th.float16, enabled=True):
                alpha_t = nn(x_t, t)
                losses = (target_t - alpha_t) ** 2
                losses = th.mean(losses.reshape(losses.shape[0], -1), dim=1)

                if direction == "bwd":
                    losses_intOperator = (
                        target_t - coarsen_field(x_0, downsample_factor=1)
                    ) ** 2

                    losses_intOperator = th.mean(
                        losses_intOperator.reshape(losses_intOperator.shape[0], -1),
                        dim=1,
                    )

                    losses = losses + losses_intOperator

                loss = th.mean(losses)

            scaler.scale(loss).backward()

            scaler.unscale_(optim)
            if grad_max_norm > 0:
                grad_norm = th.nn.utils.clip_grad_norm_(nn.parameters(), grad_max_norm)

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

                # ########################### High Resolution ###########################
                console.log("High Resolution")
                with th.no_grad():
                    te_s_path = th.zeros(
                        size=(discretization_steps + 1,) + (test_batch_dim, 1, 64, 64),
                        device=device,
                    )
                    te_p_path = th.zeros(
                        size=(discretization_steps,) + (test_batch_dim, 1, 64, 64),
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

                    for te_x_0, te_x_1 in zip(te_loader_x_0, te_loader_x_1):

                        te_x_0 = F.resize(te_x_0, 64).to(device)
                        te_x_1 = F.resize(te_x_1, 64).to(device)

                        te_s_path[0] = te_x_0
                        drift_norm.append(
                            euler_discretization(te_s_path, te_p_path, sample_nn, sigma)
                        )

                        break

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
                                    f"{direction}/test_32/x[{i}-{5}]": image_grid(
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
                                    f"{direction}/test_32/p[{i}-{5}]": image_grid(
                                        denorm_field
                                    )
                                },
                                step=step,
                            )

            if step % loss_log_steps == 0:
                if rank == 0:
                    wandb.log({f"{direction}/train/loss": loss.item()}, step=step)
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
    fn = "./checkpoint/grf_10k_256.npz"
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
    load_name = "./checkpoint/grf_10k_256.npz"
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
        "cache_batch_dim": 1024,  #
        "cache_steps": 250,
        "test_steps": 5000,  # More frequent evaluation
        "iterations": 30,
        "load": False,  # continue training
    }

    fire.Fire(run(**config))
