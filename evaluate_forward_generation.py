import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from dbfs.models.transformer import OperatorTransformer
from dbfs.dct import dct_2d, idct_2d
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import Dataset


class ToTensorCustom:
    def __call__(self, array):
        tensor = th.from_numpy(array).float()
        # Add channel dimension if missing
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # Now tensor shape is (1, H, W)
        return tensor


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


def euler_discretization(x, xp, nn, energy, chunk_size=256):
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


def load_coarse_fields(npy_path):
    all_fields = np.load(npy_path)
    return all_fields


def evaluate_convergence(
    evaluation_point, sample_sizes, coarse_field, fwd_model, coarse_normalizer, device
):
    """
    Evaluate convergence at a specific point for different sample sizes.

    Args:
        evaluation_point (tuple): (row, col) indices for evaluation
        sample_sizes (list): List of different sample sizes to test
        coarse_field (numpy.ndarray): Original coarse field (16x16)
        fwd_model: Forward model for generation
        coarse_normalizer: Normalizer for the coarse field
        device: Computation device
    """
    # Get reference value at evaluation point from original coarse field
    ref_value = coarse_field[evaluation_point]

    convergence_results = {}

    # Prepare normalized and upsampled field for model input
    normalized_field = coarse_normalizer(coarse_field)
    if normalized_field.dim() == 2:
        normalized_field = normalized_field.unsqueeze(0)
    upsampled_field = upsample(normalized_field)

    for N in sample_sizes:
        # Create batch of size N
        coarse_batch = upsampled_field.unsqueeze(0).repeat(N, 1, 1, 1).to(device)

        # Initialize paths for generation
        discretization_steps = 30
        sigma = 1.0
        s_path = th.zeros((discretization_steps + 1, N, 1, 32, 32), device=device)
        p_path = th.zeros((discretization_steps, N, 1, 32, 32), device=device)

        # Set initial condition
        s_path[0] = coarse_batch

        # Generate samples
        with th.no_grad():
            euler_discretization(s_path, p_path, fwd_model, sigma)

        generated_fine_fields = s_path[-1]

        # Coarsen generated fields
        coarsened_fields = []
        for i in range(N):
            coarsened = coarsen_field(
                generated_fine_fields[i], filter_sigma=2.0, downsample_factor=2
            )
            # Denormalize
            coarsened = coarse_normalizer.denormalize(coarsened.cpu())
            coarsened_fields.append(coarsened.squeeze(0))

        # Extract values at evaluation point
        values = [cf[evaluation_point].item() for cf in coarsened_fields]

        # Compute statistics
        convergence_results[N] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values,
        }

    return ref_value, convergence_results


def plot_convergence(ref_value, convergence_results, checkpoint_name, save_dir):
    """Plot convergence analysis results."""
    sample_sizes = sorted(convergence_results.keys())
    means = [convergence_results[N]["mean"] for N in sample_sizes]
    stds = [convergence_results[N]["std"] for N in sample_sizes]

    plt.figure(figsize=(10, 6))
    plt.errorbar(sample_sizes, means, yerr=stds, fmt="o-", label="Sample Mean ± Std")
    plt.axhline(y=ref_value, color="r", linestyle="--", label="Reference Value")

    plt.xscale("log")
    plt.xlabel("Number of Samples")
    plt.ylabel("Field Value")
    plt.title(f"Convergence Analysis - {checkpoint_name}")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(
        f"{save_dir}/convergence_{checkpoint_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def compute_field_l2_loss(
    x0,
    fwd_model,
    coarse_normalizer,
    device,
    n_samples=500,
    discretization_steps=30,
    sigma=1.0,
):
    """
    Compute the mean L2 loss between the original coarse field and the generated samples.

    Parameters:
        x0 (numpy.ndarray): The original coarse field (16x16).
        fwd_model: The forward generation network (already in eval mode).
        coarse_normalizer: Normalizer for the coarse field.
        device: Torch device (CPU or GPU).
        n_samples (int): Number of generation samples to draw (default: 500).
        discretization_steps (int): Number of steps for Euler discretization (default: 30).
        sigma (float): Energy parameter for generation (default: 1.0).

    Returns:
        mean_loss (float): The mean L2 loss computed over the n_samples.
        std_loss (float): The standard deviation of L2 losses.
        mean_relative_loss (float): The mean relative L2 loss.
        std_relative_loss (float): The standard deviation of relative L2 losses.
    """
    # Prepare normalized and upsampled field for model input
    normalized_field = coarse_normalizer(x0)
    if normalized_field.dim() == 2:
        normalized_field = normalized_field.unsqueeze(0)
    upsampled_field = upsample(normalized_field)

    # Create batch of size n_samples
    coarse_batch = upsampled_field.unsqueeze(0).repeat(n_samples, 1, 1, 1).to(device)

    # Initialize paths for generation
    s_path = th.zeros((discretization_steps + 1, n_samples, 1, 32, 32), device=device)
    p_path = th.zeros((discretization_steps, n_samples, 1, 32, 32), device=device)

    # Set initial condition
    s_path[0] = coarse_batch

    # Generate samples
    with th.no_grad():
        euler_discretization(s_path, p_path, fwd_model, sigma)

    generated_fine_fields = s_path[-1]

    # Compute original field norm for relative loss
    x0_norm = th.norm(th.tensor(x0)).item()

    # Coarsen generated fields and compute losses
    losses = []
    relative_losses = []
    for i in range(n_samples):
        # Coarsen the generated fine field
        coarsened = coarsen_field(
            generated_fine_fields[i], filter_sigma=2.0, downsample_factor=2
        )
        # Denormalize
        coarsened = coarse_normalizer.denormalize(coarsened.cpu())

        # Compute absolute L2 loss
        loss = th.norm(coarsened.squeeze(0) - th.tensor(x0)).item()
        losses.append(loss)

        # Compute relative L2 loss
        relative_loss = loss / x0_norm
        relative_losses.append(relative_loss)

    return (
        np.mean(losses),
        np.std(losses),
        np.mean(relative_losses),
        np.std(relative_losses),
    )


def compute_mean_l2_loss_over_test_set(
    test_dataset, fwd_model, coarse_normalizer, device, n_samples=500
):
    """
    Compute the overall mean L2 loss over all coarse fields in the test set.

    Parameters:
        test_dataset (FieldsDataset): Dataset of coarse fields.
        fwd_model: The forward generation network.
        coarse_normalizer: Normalizer for the coarse field.
        device: Torch device.
        n_samples (int): Number of generation samples per test field (default: 500).

    Returns:
        overall_mean_loss (float): Mean L2 loss averaged over the entire test set.
        overall_std_loss (float): Standard deviation of L2 losses over the test set.
        overall_mean_relative_loss (float): Mean relative L2 loss over the test set.
        overall_std_relative_loss (float): Standard deviation of relative L2 losses.
        per_field_results (list): List of (mean, std, rel_mean, rel_std) tuples for each test field.
    """
    per_field_results = []

    print(f"Computing L2 losses for {len(test_dataset)} test fields...")
    # draw 500 random fields from the test set
    random_indices = np.random.choice(len(test_dataset), size=500, replace=False)
    for i in random_indices:
        # Get original coarse field (without normalization/upsampling)
        x0 = test_dataset.data[i]

        mean_loss, std_loss, mean_rel_loss, std_rel_loss = compute_field_l2_loss(
            x0, fwd_model, coarse_normalizer, device, n_samples
        )
        per_field_results.append((mean_loss, std_loss, mean_rel_loss, std_rel_loss))

        print(
            f"Field {i+1}/{len(test_dataset)}: "
            f"Mean L2 loss = {mean_loss:.4f} ± {std_loss:.4f}, "
            f"Mean Relative L2 loss = {mean_rel_loss:.4f} ± {std_rel_loss:.4f}"
        )

    # Compute overall statistics
    field_means = [r[0] for r in per_field_results]
    field_rel_means = [r[2] for r in per_field_results]

    overall_mean_loss = np.mean(field_means)
    overall_std_loss = np.std(field_means)
    overall_mean_relative_loss = np.mean(field_rel_means)
    overall_std_relative_loss = np.std(field_rel_means)

    return (
        overall_mean_loss,
        overall_std_loss,
        overall_mean_relative_loss,
        overall_std_relative_loss,
        per_field_results,
    )


def check_existing_evaluation(checkpoint_name, output_dir):
    """Check if evaluation results already exist for the checkpoint."""
    results_path = f"{output_dir}/l2_losses_{checkpoint_name}.npy"
    if os.path.exists(results_path):
        print(f"Found existing evaluation results for {checkpoint_name}")
        return np.load(results_path, allow_pickle=True).item()
    return None


def plot_generation_samples(
    original_coarse,
    true_fine,
    generated_fine_fields,
    coarse_normalizer,
    fine_normalizer,
    checkpoint_name,
    output_dir,
    num_examples=3,
):
    """
    Plot two figures:
    1. Reference fields: Original coarse, true fine, mean generated fine, and mean coarsened
    2. Generated samples: Fine fields and their coarsened counterparts
    """
    # Compute mean of generated fields
    mean_generated = generated_fine_fields.mean(dim=0)
    mean_generated_np = (
        fine_normalizer.denormalize(mean_generated.cpu()).squeeze(0).numpy()
    )

    # Compute coarsened mean
    coarsened_mean = (
        coarsen_field(
            th.tensor(mean_generated_np).unsqueeze(0),
            filter_sigma=2.0,
            downsample_factor=2,
        )
        .squeeze(0)
        .numpy()
    )

    # Figure 1: Reference Fields
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))

    # Original coarse field
    im0 = axes1[0, 0].imshow(original_coarse, cmap="viridis")
    axes1[0, 0].set_title("Original Coarse Field")
    fig1.colorbar(im0, ax=axes1[0, 0])

    # True fine field
    im1 = axes1[0, 1].imshow(true_fine, cmap="viridis")
    axes1[0, 1].set_title("True Fine Field")
    fig1.colorbar(im1, ax=axes1[0, 1])

    # Mean generated fine field
    im2 = axes1[1, 0].imshow(mean_generated_np, cmap="viridis")
    axes1[1, 0].set_title("Mean Generated Fine Field")
    fig1.colorbar(im2, ax=axes1[1, 0])

    # Mean coarsened field
    im3 = axes1[1, 1].imshow(coarsened_mean, cmap="viridis")
    axes1[1, 1].set_title("Mean Coarsened Field")
    fig1.colorbar(im3, ax=axes1[1, 1])

    plt.suptitle(f"Reference Fields - Model: {checkpoint_name}")
    plt.tight_layout()

    # Save reference fields figure
    plt.savefig(
        f"{output_dir}/reference_fields_{checkpoint_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Figure 2: Generated Samples
    fig2, axes2 = plt.subplots(2, num_examples, figsize=(15, 8))

    # Process and plot generated samples
    for i in range(num_examples):
        # Get and process fine sample
        fine_sample = generated_fine_fields[i]
        fine_sample = fine_normalizer.denormalize(fine_sample.cpu())
        fine_np = fine_sample.squeeze(0).numpy()

        # Get coarsened sample
        coarsened = coarsen_field(fine_sample, filter_sigma=2.0, downsample_factor=2)
        coarsened_np = coarsened.squeeze(0).numpy()

        # Plot fine sample
        im_fine = axes2[0, i].imshow(fine_np, cmap="viridis")
        axes2[0, i].set_title(f"Generated Fine {i+1}")
        fig2.colorbar(im_fine, ax=axes2[0, i])

        # Plot coarsened sample
        im_coarse = axes2[1, i].imshow(coarsened_np, cmap="viridis")
        axes2[1, i].set_title(f"Coarsened {i+1}")
        fig2.colorbar(im_coarse, ax=axes2[1, i])

    plt.suptitle(f"Generated Samples - Model: {checkpoint_name}")
    plt.tight_layout()

    # Save generated samples figure
    plt.savefig(
        f"{output_dir}/generated_samples_{checkpoint_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    # Setup
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Define checkpoints to evaluate
    checkpoints = ["grf_10k_256_optimized.npz", "grf_10k_256_intOp_scaled.npz"]

    # Create output directory
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.replace(".npz", "")
        checkpoint_path = f"./dbfs/checkpoint/{checkpoint}"
        coarse_data_path = "./Data/coarse_grf_10k.npy"
        fine_data_path = "./Data/fine_grf_10k.npy"

        # Check for existing evaluation results
        existing_results = check_existing_evaluation(checkpoint_name, output_dir)

        if existing_results is None:
            # Load normalizer and model
            coarse_normalizer = DatasetNormalization(coarse_data_path)
            fwd_model = load_forward_model(checkpoint_path, device)

            # Create test dataset
            te_data_0 = FieldsDataset(
                data_path=coarse_data_path,
                train=False,
                transform=None,
            )

            # Compute L2 losses over test set
            print(f"\nEvaluating L2 losses for checkpoint: {checkpoint}")
            results = compute_mean_l2_loss_over_test_set(
                te_data_0, fwd_model, coarse_normalizer, device
            )

            # Save results
            (
                overall_mean,
                overall_std,
                overall_rel_mean,
                overall_rel_std,
                per_field_results,
            ) = results
            results_dict = {
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "overall_relative_mean": overall_rel_mean,
                "overall_relative_std": overall_rel_std,
                "per_field_results": per_field_results,
            }
            np.save(f"{output_dir}/l2_losses_{checkpoint_name}.npy", results_dict)
        else:
            results_dict = existing_results
            print(f"\nUsing existing evaluation results for: {checkpoint}")

        # Print results
        print(
            f"Overall L2 loss: {results_dict['overall_mean']:.4f} ± {results_dict['overall_std']:.4f}\n"
            f"Overall Relative L2 loss: {results_dict['overall_relative_mean']:.4f} ± {results_dict['overall_relative_std']:.4f}"
        )

        # Generate visualization samples
        coarse_normalizer = DatasetNormalization(coarse_data_path)
        fine_normalizer = DatasetNormalization(fine_data_path)
        fwd_model = load_forward_model(checkpoint_path, device)

        # Load test datasets
        te_data_0 = FieldsDataset(
            data_path=coarse_data_path, train=False, transform=None
        )
        te_data_1 = FieldsDataset(data_path=fine_data_path, train=False, transform=None)

        # Get corresponding coarse and fine fields
        original_coarse = te_data_0.data[0]
        true_fine = te_data_1.data[0]

        # Prepare input for generation
        normalized_field = coarse_normalizer(original_coarse)
        if normalized_field.dim() == 2:
            normalized_field = normalized_field.unsqueeze(0)
        upsampled_field = upsample(normalized_field)

        # Generate samples
        batch_size = 16
        coarse_field_batch = (
            upsampled_field.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        )

        discretization_steps = 30
        sigma = 1.0
        s_path = th.zeros(
            (discretization_steps + 1, batch_size, 1, 32, 32), device=device
        )
        p_path = th.zeros((discretization_steps, batch_size, 1, 32, 32), device=device)

        s_path[0] = coarse_field_batch

        with th.no_grad():
            euler_discretization(s_path, p_path, fwd_model, sigma)

        generated_fine_fields = s_path[-1]

        # Plot samples with true fine field
        plot_generation_samples(
            original_coarse,
            true_fine,
            generated_fine_fields,
            coarse_normalizer,
            fine_normalizer,
            checkpoint_name,
            output_dir,
        )


if __name__ == "__main__":
    main()
