import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from dbfs.models.transformer import OperatorTransformer
from dbfs.dct import dct_2d, idct_2d
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel as DDP


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


# Reuse coarsening function from training script
def coarsen_field(field, filter_sigma=2.0, downsample_factor=2, method="bilinear"):
    if not isinstance(field, th.Tensor):
        raise ValueError("Field must be a torch tensor")

    if field.dim() == 3:
        field = field.unsqueeze(0)
        squeeze = True
    elif field.dim() == 4:
        squeeze = False
    else:
        raise ValueError("Field must have 3 or 4 dimensions")

    kernel_size = 2 * int(4 * filter_sigma + 0.5) + 1
    smooth = TF.gaussian_blur(
        field,
        kernel_size=(kernel_size, kernel_size),
        sigma=(filter_sigma, filter_sigma),
    )

    scale_factor = 1.0 / downsample_factor
    coarse = th.nn.functional.interpolate(
        smooth, scale_factor=scale_factor, mode=method, align_corners=False
    )

    if squeeze:
        coarse = coarse.squeeze(0)
    return coarse


def upsample(field):
    field = F.interpolate(
        field.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
    ).squeeze(0)
    return field


class DatasetNormalization:
    def __init__(self, train_data_path):
        train_data = np.load(train_data_path)
        self.mean = train_data.mean()
        self.std = train_data.std()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = th.from_numpy(x).float()
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


def load_forward_model(checkpoint_path, device):
    # Initialize all components that were saved
    bwd_nn = OperatorTransformer(
        in_channel=2,
        out_channel=1,
        latent_dim=256,
        pos_dim=256,
        num_heads=4,
        depth_enc=6,
        depth_dec=2,
        scale=1,
        self_per_cross_attn=1,
        height=32,
    ).to(device)

    fwd_nn = OperatorTransformer(
        in_channel=2,
        out_channel=1,
        latent_dim=256,
        pos_dim=256,
        num_heads=4,
        depth_enc=6,
        depth_dec=2,
        scale=1,
        self_per_cross_attn=1,
        height=32,
    ).to(device)

    # Initialize EMA helpers
    bwd_ema = EMAHelper(bwd_nn, mu=0.999, device=device)
    fwd_ema = EMAHelper(fwd_nn, mu=0.999, device=device)

    # Initialize optimizers
    bwd_optim = th.optim.Adam(bwd_nn.parameters(), lr=1e-4)
    fwd_optim = th.optim.Adam(fwd_nn.parameters(), lr=1e-4)

    # Create list of components to load
    saves = [bwd_nn, bwd_ema, bwd_optim, fwd_nn, fwd_ema, fwd_optim]

    # Load checkpoint
    checkpoint = th.load(checkpoint_path, map_location=device)

    # Load state dicts for all components
    for i, key in enumerate(
        ["bwd_nn", "bwd_ema", "bwd_optim", "fwd_nn", "fwd_ema", "fwd_optim"]
    ):
        saves[i].load_state_dict(checkpoint[key])

    # Create forward sampling network (EMA copy)
    fwd_sample_nn = fwd_ema.ema_copy()
    fwd_sample_nn.eval()

    return fwd_sample_nn


def load_and_prepare_coarse_field(npy_path, normalizer, device, batch_size=16):
    all_fields = np.load(npy_path)
    field = all_fields[0]  # Take first sample
    field = th.from_numpy(field).float()
    if field.dim() == 2:
        field = field.unsqueeze(0)

    # Normalize and upsample
    field = normalizer(field)
    field = upsample(field)

    # Expand to batch
    field_batch = field.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    return field_batch, field


def main():
    # Setup
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # without integral operator loss, and a scaled integral operator loss
    checkpoints = ["grf_10k_256_optimized.npz", "grf_10k_256_intOp_scaled.npz"]

    for checkpoint in checkpoints:
        checkpoint_path = f"./dbfs/checkpoint/{checkpoint}"
        coarse_data_path = f"./Data/coarse_grf_10k.npy"

    # Load normalizer
    coarse_normalizer = DatasetNormalization(coarse_data_path)

    # Load model
    fwd_model = load_forward_model(checkpoint_path, device)

    # Prepare input
    batch_size = 16
    coarse_field_batch, original_coarse = load_and_prepare_coarse_field(
        coarse_data_path, coarse_normalizer, device, batch_size
    )

    # Setup SDE parameters
    discretization_steps = 30
    sigma = 1.0

    # Initialize paths
    s_path = th.zeros((discretization_steps + 1, batch_size, 1, 32, 32), device=device)
    p_path = th.zeros((discretization_steps, batch_size, 1, 32, 32), device=device)

    # Set initial condition
    s_path[0] = coarse_field_batch

    # Generate samples
    with th.no_grad():
        euler_discretization(s_path, p_path, fwd_model, sigma)

    generated_fine_fields = s_path[-1]

    # Compute mean
    mean_generated = generated_fine_fields.mean(dim=0)

    # Select and coarsen examples
    num_examples = 3
    coarsened_examples = []
    for i in range(num_examples):
        sample = generated_fine_fields[i]
        coarsened = coarsen_field(sample, filter_sigma=2.0, downsample_factor=2)
        # Denormalize
        coarsened = coarse_normalizer.denormalize(coarsened.cpu())
        coarsened_examples.append(coarsened.squeeze(0).numpy())

    # Process original and mean for plotting
    original_coarse = (
        coarse_normalizer.denormalize(original_coarse.cpu()).squeeze(0).numpy()
    )
    mean_generated = (
        coarse_normalizer.denormalize(mean_generated.cpu()).squeeze(0).numpy()
    )

    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Plot original coarse field
    im0 = axes[0].imshow(original_coarse, cmap="viridis")
    axes[0].set_title("Original Coarse Field")
    fig.colorbar(im0, ax=axes[0])

    # Plot mean of generated fine fields
    im1 = axes[1].imshow(mean_generated, cmap="viridis")
    axes[1].set_title("Mean Generated Fine Field")
    fig.colorbar(im1, ax=axes[1])

    # Plot coarsened examples
    for idx in range(num_examples):
        im = axes[idx + 2].imshow(coarsened_examples[idx], cmap="viridis")
        axes[idx + 2].set_title(f"Coarsened Sample {idx+1}")
        fig.colorbar(im, ax=axes[idx + 2])

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs("evaluation_results", exist_ok=True)
    plt.savefig(
        "evaluation_results/forward_generation_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
