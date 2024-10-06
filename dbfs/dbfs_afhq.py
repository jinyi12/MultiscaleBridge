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
warnings.filterwarnings('ignore')

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
from torch.utils.data import DataLoader, Subset
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
    frequencies_squared = (freq + 1.)

    a_k = frequencies_squared[None, None]
    Sigma_k = th.pow(a_k, -0.02) / (energy**2)

    denominator = (v_(t_start, t, a_k) * a_2(t, t_end, a_k) + v_(t, t_end, a_k))
    coeff_1 = (v_(t, t_end, a_k) * a_(t_start, t, a_k)) / denominator
    coeff_2 = (v_(t_start, t, a_k) * a_(t, t_end, a_k)) / denominator
    coeff_3 = (v_(t_start, t, a_k) * v_(t, t_end, a_k)) / denominator
    
    x_0 = dct_2d(x_0, norm='ortho')
    x_1 = dct_2d(x_1, norm='ortho')
    
    mean_t = coeff_1 * x_0 + coeff_2 * x_1
    var_t = coeff_3 * Sigma_k
    z_t = dct_2d(th.randn_like(x_0), norm='ortho')
    
    x_t = mean_t + th.sqrt(var_t) * z_t
    x_t = idct_2d(x_t, norm='ortho')
    
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
    frequencies_squared = (freq + 1.)
    a_k = frequencies_squared[None, None]
    sigma_k = th.pow(a_k, -0.01) / energy
    
    for i in range(1, T + 1):
        t = dt * (i - 1)
        alpha_t = nn(x[i - 1], t)
        drift_norms = drift_norms + th.mean(alpha_t.reshape(B, -1) ** 2, dim=1)
        
        alpha_t = dct_2d(alpha_t, norm='ortho')
        x_i = dct_2d(x[i - 1], norm='ortho')
        t_ = t[:, None, None, None]
        t_end = th.ones_like(t_)
        xp_coeff = alpha_t
        xp[i - 1] = idct_2d(xp_coeff, norm='ortho')
        control = (a_(t_, t_end, a_k) * alpha_t - a_2(t_, t_end, a_k) * x_i) / v_(t_, t_end, a_k)
        drift_t = (-a_k * x_i + control) * dt[:, None, None, None]
        eps_t = dct_2d(th.randn_like(x[i - 1]), norm='ortho')
    
        if i == T:
            diffusion_t = 0
        else:
            diffusion_t = sigma_k * th.sqrt(dt[:, None, None, None]) * eps_t
        x[i] = idct_2d(x_i + drift_t + diffusion_t, norm='ortho')
        
    drift_norms = drift_norms / T
    return drift_norms.cpu()


# Data ---------------------------------------------------------------------------------

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

# Linear scaling to float [-1.0, 1.0]:

transform=transforms.Compose(
    [ #Normal Data preprocessing
    transforms.Resize([64 , 64]), #DSB type resizing
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
])

tr_dataset= datasets.ImageFolder(root='./data/afhq/train', transform=transform)

tr_idx_0 = [i for i in range(len(tr_dataset)) if tr_dataset.imgs[i][1] == tr_dataset.class_to_idx['cat']]
tr_idx_1 = [i for i in range(len(tr_dataset)) if tr_dataset.imgs[i][1] == tr_dataset.class_to_idx['dog']]

tr_data_0 = Subset(tr_dataset, tr_idx_0)
tr_data_1 = Subset(tr_dataset, tr_idx_1)

te_dataset= datasets.ImageFolder(root='./data/afhq/val', transform=transform)

te_idx_0 = [i for i in range(len(te_dataset)) if te_dataset.imgs[i][1] == te_dataset.class_to_idx['cat']]
te_idx_1 = [i for i in range(len(te_dataset)) if te_dataset.imgs[i][1] == te_dataset.class_to_idx['dog']]

te_data_0 = Subset(te_dataset, te_idx_0)
te_data_1 = Subset(te_dataset, te_idx_1)

te_data_0 = Subset(te_data_0, rng.permutation(len(te_data_0)))
te_data_1 = Subset(te_data_1, rng.permutation(len(te_data_1)))


# DDP
tr_spl_0 = DistributedSampler(tr_data_0, shuffle=True) if parallel else None
tr_spl_1 = DistributedSampler(tr_data_1, shuffle=True) if parallel else None
te_spl_0 = DistributedSampler(te_data_0, shuffle=False) if parallel else None
te_spl_1 = DistributedSampler(te_data_1, shuffle=False) if parallel else None

# NN Model -----------------------------------------------------------------------------


def init_nn():
    return OperatorTransformer(
        in_channel = 2,
        out_channel = 3,
        latent_dim = 512,
        pos_dim = 512,
        num_heads = 4,
        depth_enc = 6,
        depth_dec = 2,
        scale = 1,
        self_per_cross_attn = 2,
        height=64
    ).to(device)
    
class EMAHelper:
    # Simplified from https://github.com/ermongroup/ddim/blob/main/models/ema.py:
    def __init__(self, module, mu=0.999, device=None):
        self.module = module.module if isinstance(module, DDP) else module # DDP
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
    iterations=20,
    training_steps=20000,
    discretization_steps=100,
    batch_dim=64,
    learning_rate=1e-4,
    grad_max_norm=1.0,
    ema_decay=0.999,
    cache_steps=1000,
    cache_batch_dim=2560,
    test_steps=20000,
    test_batch_dim=256, 
    loss_log_steps=100,
    imge_log_steps=1000,
    load=False
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
        console = console,
        speed_estimate_period = 60 * 5,
    )
    iteration_t = progress.add_task("iteration", total=iterations)
    step_t = progress.add_task("step", total=iterations * training_steps)

    if rank == 0: wandb.init(project = "dbfs", config=config, mode="online")
    if rank == 0: console.log(wandb.config)

    # DDP
    if parallel:
        world_size = dist.get_world_size()
        if batch_dim % world_size != 0:
            raise ValueError(f"minibatch_size ({batch_dim}) must be divisible by world_size ({world_size})")
        batch_dim = batch_dim // world_size
        if cache_batch_dim % world_size != 0:
            raise ValueError(f"cache_batch_dim ({cache_batch_dim}) must be divisible by world_size ({world_size})")
        cache_batch_dim = cache_batch_dim // world_size
        if test_batch_dim % world_size != 0:
            raise ValueError(f"test_batch_dim ({test_batch_dim}) must be divisible by world_size ({world_size})")
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

    saves = [bwd_nn, bwd_ema, bwd_optim,
             fwd_nn, fwd_ema, fwd_optim]
    
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
        
    if rank == 0: console.log(f"Training Start from Iteration : {start_iteration}")

    dt = 1.0 / discretization_steps
    t_T = 1.0 - dt * 0.5

    s_path = th.zeros(
        size=(discretization_steps + 1,) + (cache_batch_dim, 3, 64, 64), device=device
    )  # i: 0, ..., discretization_steps;     t: 0, dt, ..., 1.0.
    p_path = th.zeros(
        size=(discretization_steps,) + (cache_batch_dim, 3, 64, 64), device=device
    )  # i: 0, ..., discretization_steps - 1; t: 0, dt, ..., 1.0 - dt.

    scaler = th.cuda.amp.GradScaler(enabled=True)
    
    if rank == 0: progress.start()
    for iteration in range(start_iteration + 1, iterations + 1):
        if rank == 0: console.log(f"iteration {iteration}: {step}")
        if rank == 0: progress.update(iteration_t, completed=iteration)
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
                    x_0 = next(tr_iter_1)[0].to(device)
                    x_1 = next(tr_iter_0)[0].to(device)
                else:
                    with th.no_grad():
                        if (step - 1) % cache_steps == 0:
                            if rank == 0: console.log(f"cache update: {step}")
                            # Simulate previously inferred SDE:
                            cache_small_batch_dim = cache_batch_dim // 10
                            
                            s_0 = next(tr_cache_iter_0)[0].to(device)
                            
                            for i in range(0, 10):
                                small_s_path = th.zeros(
                                    size=(discretization_steps + 1,) + (cache_small_batch_dim, 3, 64, 64), device=device
                                )  # i: 0, ..., discretization_steps;     t: 0, dt, ..., 1.0.
                                small_p_path = th.zeros(
                                    size=(discretization_steps,) + (cache_small_batch_dim, 3, 64, 64), device=device
                                )  # i: 0, ..., discretization_steps - 1; t: 0, dt, ..., 1.0 - dt.

                                small_s_path[0] = s_0[i*cache_small_batch_dim : (i+1)*cache_small_batch_dim]
                                euler_discretization(small_s_path, small_p_path, fwd_sample_nn, sigma)
                                
                                s_path[:, i*cache_small_batch_dim : (i+1)*cache_small_batch_dim] = small_s_path
                                p_path[:, i*cache_small_batch_dim : (i+1)*cache_small_batch_dim] = small_p_path

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
                        if rank == 0: console.log(f"cache update: {step}")
                        # Simulate previously inferred SDE:
                        
                        cache_small_batch_dim = cache_batch_dim // 10

                        s_1 = next(tr_cache_iter_1)[0].to(device)

                        for i in range(0, 10):
                            small_s_path = th.zeros(
                                size=(discretization_steps + 1,) + (cache_small_batch_dim, 3, 64, 64), device=device
                            )  # i: 0, ..., discretization_steps;     t: 0, dt, ..., 1.0.
                            small_p_path = th.zeros(
                                size=(discretization_steps,) + (cache_small_batch_dim, 3, 64, 64), device=device
                            )  # i: 0, ..., discretization_steps - 1; t: 0, dt, ..., 1.0 - dt.

                            small_s_path[0] = s_1[i*cache_small_batch_dim : (i+1)*cache_small_batch_dim]
                            euler_discretization(small_s_path, small_p_path, bwd_sample_nn, sigma)
                            
                            s_path[:, i*cache_small_batch_dim : (i+1)*cache_small_batch_dim] = small_s_path
                            p_path[:, i*cache_small_batch_dim : (i+1)*cache_small_batch_dim] = small_p_path

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

            with th.autocast(device_type='cuda', dtype=th.float16, enabled=True):
                alpha_t = nn(x_t, t)
                losses = (target_t - alpha_t) ** 2
                losses = th.mean(losses.reshape(losses.shape[0], -1), dim=1)
                loss = th.mean(losses)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optim)
            if grad_max_norm > 0:
                grad_norm = th.nn.utils.clip_grad_norm_(nn.parameters(), grad_max_norm)
                
            scaler.step(optim)
            scaler.update()
            ema.update()

            if step % test_steps == 0:
                if rank == 0: console.log(f"test: {step}")
                ema.ema(sample_nn)

                # Save per each iteration
                saves = [bwd_nn, bwd_ema, bwd_optim,
                         fwd_nn, fwd_ema, fwd_optim]
                save_checkpoint(saves, iteration, console)
                
                with th.no_grad():
                    te_s_path = th.zeros(
                        size=(discretization_steps + 1,) + (test_batch_dim, 3, 64, 64),
                        device=device,
                    )
                    te_p_path = th.zeros(
                        size=(discretization_steps,) + (test_batch_dim, 3, 64, 64),
                        device=device,
                    )
                    # Assumes data is in [0.0, 1.0], scale appropriately:
                    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
                    drift_norm = []
                    
                    for te_x_0, te_x_1 in zip(te_loader_x_0, te_loader_x_1):
                        te_x_0, te_x_1 = te_x_0[0].to(device), te_x_1[0].to(device)
                        te_x_1 = (te_x_1 + 1.0) / 2.0
                        te_s_path[0] = te_x_0
                        drift_norm.append(
                            euler_discretization(te_s_path, te_p_path, sample_nn, sigma)
                        )
                        te_s_path = th.clip((te_s_path + 1.0) / 2.0, 0.0, 1.0)
                        te_p_path = th.clip((te_p_path + 1.0) / 2.0, 0.0, 1.0)
                        fid_metric.update(te_x_1, real=True)
                        fid_idx = -2
                        fid_metric.update(
                            te_p_path[fid_idx], real=False
                        )
                    drift_norm = th.mean(th.cat(drift_norm)).item()
                    fid = fid_metric.compute().item()
                    if rank == 0: wandb.log({f"{direction}/test/drift_norm": drift_norm}, step=step)
                    if rank == 0: wandb.log({f"{direction}/test/fid": fid}, step=step)
                    if rank == 0: console.log(f"FID: {fid}")
                    for i, ti in enumerate(
                        resample_indices(discretization_steps + 1, 5)
                    ):
                        if rank == 0: wandb.log(
                            {f"{direction}/test/x[{i}-{5}]": image_grid(te_s_path[ti])},
                            step=step,
                        )
                    for i, ti in enumerate(resample_indices(discretization_steps, 5)):
                        if rank == 0: wandb.log(
                            {f"{direction}/test/p[{i}-{5}]": image_grid(te_p_path[ti])},
                            step=step,
                        )
                    
                    del te_s_path
                    del te_p_path
                # ########################### High Resolution ###########################
                with th.no_grad():
                    te_s_path = th.zeros(
                        size=(discretization_steps + 1,) + (2, 3, 128, 128),
                        device=device,
                    )
                    te_p_path = th.zeros(
                        size=(discretization_steps,) + (2, 3, 128, 128),
                        device=device,
                    )
                    # Assumes data is in [0.0, 1.0], scale appropriately:
                    drift_norm = []
                    for te_x_0, te_x_1 in zip(te_loader_x_0, te_loader_x_1):
                        
                        te_x_0 = F.resize(te_x_0[0], 128).to(device)[:2]
                        te_x_1 = F.resize(te_x_1[0], 128).to(device)[:2]

                        te_x_1 = (te_x_1 + 1.0) / 2.0
                        te_s_path[0] = te_x_0
                        drift_norm.append(
                            euler_discretization(te_s_path, te_p_path, sample_nn, sigma)
                        )
                        te_s_path = th.clip((te_s_path + 1.0) / 2.0, 0.0, 1.0)
                        te_p_path = th.clip((te_p_path + 1.0) / 2.0, 0.0, 1.0)    
                        
                        break 
                    
                    for i, ti in enumerate(
                        resample_indices(discretization_steps + 1, 5)
                    ):
                        if rank == 0: wandb.log(
                            {f"{direction}/test_128/x[{i}-{5}]": image_grid(te_s_path[ti])},
                            step=step,
                        )
                    for i, ti in enumerate(resample_indices(discretization_steps, 5)):
                        if rank == 0: wandb.log(
                            {f"{direction}/test_128/p[{i}-{5}]": image_grid(te_p_path[ti])},
                            step=step,
                        )
                    del te_s_path
                    del te_p_path             
                    
            if step % loss_log_steps == 0:
                if rank == 0: wandb.log({f"{direction}/train/loss": loss.item()}, step=step)
                if rank == 0: wandb.log({f"{direction}/train/grad_norm": grad_norm}, step=step)

            if step % imge_log_steps == 0:
                if rank == 0: wandb.log({f"{direction}/train/x_0": image_grid(x_0, True)}, step=step)
                if rank == 0: wandb.log({f"{direction}/train/x_1": image_grid(x_1, True)}, step=step)

            if step % training_steps == 0:
                if rank == 0: console.log(f"EMA update: {step}")
                # Make sure EMA is updated at the end of each iteration:
                ema.ema(sample_nn)

    progress.stop()

def save_checkpoint(saves, iteration, console):
    # Simplified from https://github.com/ghliu/SB-FBSDE/blob/main/util.py:
    checkpoint = {}
    i = 0
    fn = './checkpoint/afhq.npz'
    keys = ['bwd_nn', 'bwd_ema', 'bwd_optim', 
            'fwd_nn', 'fwd_ema', 'fwd_optim']
    
    with th.cuda.device(rank):
        for k in keys:
            checkpoint[k] = saves[i].state_dict()
            i += 1
        checkpoint['iteration'] = iteration
        th.save(checkpoint, fn)
    if rank == 0: console.log(f"checkpoint saved: {fn}")
    

def restore_checkpoint(saves, console):
    # Simplified from https://github.com/ghliu/SB-FBSDE/blob/main/util.py:
    i = 0
    load_name = './checkpoint/afhq.npz'
    assert load_name is not None
    if rank == 0: console.log(f"#loading checkpoint {load_name}...")

    with th.cuda.device(rank):
        checkpoint = th.load(load_name, map_location=th.device('cuda:%d' % rank))
        ckpt_keys=[*checkpoint.keys()][:-1]
        for k in ckpt_keys:
            saves[i].load_state_dict(checkpoint[k])
            i += 1
    if rank == 0: console.log('#successfully loaded all the modules')
        
    return checkpoint['iteration']


if __name__ == "__main__":
    fire.Fire(run)
