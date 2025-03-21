import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from dbfs.models.transformer import OperatorTransformer
from dbfs.models.bm2_transformer import BM2Transformer
from dbfs.dct import dct_2d, idct_2d
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import Dataset
from ema_pytorch import EMA
from dbfs.utils import make_data_grid
from einops import repeat, rearrange

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


def load_model_from_wandb(wandb_project, wandb_name, version="latest", device=None):
    """
    Load a model from a wandb artifact with specific version.

    Args:
        wandb_project (str): The wandb project path (e.g., "jyyresearch/BM2_GRF_MultiscaleBridge")
        wandb_name (str): The wandb run name or model name
        version (str): Artifact version (e.g., "latest", "v0", "v1", etc.)
        device (torch.device): The device to load the model on

    Returns:
        nn: The loaded model ready for evaluation
    """
    import wandb

    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Initialize wandb run
    run = wandb.init(project=wandb_project.split("/")[-1], job_type="evaluation")

    # Use the artifact with specified version
    artifact_path = f"{wandb_project}/{wandb_name}_model_checkpoint:{version}"
    print(f"Loading wandb artifact: {artifact_path}")

    artifact = run.use_artifact(artifact_path, type="model_checkpoint")
    artifact_dir = artifact.download()

    # Find the checkpoint file in the downloaded directory
    checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith(".npz")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {artifact_dir}")

    checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
    print(f"Found checkpoint: {checkpoint_path}")

    # Now use the existing function to load the model from the downloaded checkpoint
    nn = load_forward_model(checkpoint_path, device)

    return nn


def load_forward_model(checkpoint_path, device):
    """
    Load the forward generation model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file saved by BM2_dbfs_grf_256_intOp_optimized_3.py
        device (torch.device): The device to load the model on

    Returns:
        nn: The loaded model that can be used for forward generation
    """
    # Initialize BM2Transformer model - make sure parameters match what was used in training
    nn = BM2Transformer(
        in_axis=2,
        out_axis=2,
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
        use_checkpointing=True,
    ).to(device)

    nn = th.compile(nn)

    # Initialize EMA helper with the model
    ema = EMA(nn, beta=0.999, update_after_step=0, update_every=1, power=3 / 4)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=device)

    # Load state dicts for all components
    nn.load_state_dict(checkpoint["nn"])
    ema.load_state_dict(checkpoint["ema"])

    # Set model to evaluation mode
    nn.eval()

    # Use the EMA weights for inference
    # This is equivalent to what the training script does with ema.update_model_with_ema()
    ema.ema_model = nn  # Set the target model for EMA to be the loaded model
    ema.update_model_with_ema()

    print("Model loaded successfully")
    return nn


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
            euler_discretization(
                x=s_path,
                nn=fwd_model,
                energy=sigma,
                chunk_size=min(N, 128),  # Use appropriately sized chunks
                store_control=False,
                direction_fn="fwd",  # Use forward head of the model
            )

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
    model,
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
        model: The neural network containing both forward and backward heads (already in eval mode).
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
        euler_discretization(
            x=s_path,
            nn=model,
            energy=sigma,
            # chunk_size=min(n_samples, 128),  # Use reasonable chunk size
            chunk_size=n_samples,  # Use reasonable chunk size
            store_control=False,
            direction_fn="fwd",  # Use forward head of the model
        )

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
        th.mean(th.tensor(losses)).item(),
        th.std(th.tensor(losses)).item(),
        th.mean(th.tensor(relative_losses)).item(),
        th.std(th.tensor(relative_losses)).item(),
    )


def compute_mean_l2_loss_over_test_set(
    test_dataset,
    model,
    coarse_normalizer,
    device,
    n_samples=500,
    n_fields=500,
    discretization_steps=30,
    energy=1.0,
):
    """
    Compute the overall mean L2 loss over all coarse fields in the test set.

    Parameters:
        test_dataset (FieldsDataset): Dataset of coarse fields.
        model: The neural network containing both forward and backward heads (already in eval mode).
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
    random_indices = np.random.choice(len(test_dataset), size=n_fields, replace=False)
    for nth_field, random_index in enumerate(random_indices):
        # Get original coarse field (without normalization/upsampling)
        x0 = test_dataset.data[random_index]

        mean_loss, std_loss, mean_rel_loss, std_rel_loss = compute_field_l2_loss(
            x0,
            model,
            coarse_normalizer,
            device,
            n_samples,
            discretization_steps,
            energy,
        )
        per_field_results.append((mean_loss, std_loss, mean_rel_loss, std_rel_loss))

        print(
            f"Field {nth_field + 1}/{len(test_dataset)}: "
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
        axes2[0, i].set_title(f"Generated Fine {i + 1}")
        fig2.colorbar(im_fine, ax=axes2[0, i])

        # Plot coarsened sample
        im_coarse = axes2[1, i].imshow(coarsened_np, cmap="viridis")
        axes2[1, i].set_title(f"Coarsened {i + 1}")
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


def main(args):
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    artifact_version = args.artifact_version
    coarse_data_path = args.coarse_data_path
    fine_data_path = args.fine_data_path
    output_dir = args.output_dir
    force_eval = args.force_eval
    batch_size = args.batch_size
    n_samples = args.n_samples
    n_fields = args.n_fields
    discretization_steps = args.discretization_steps
    energy = args.energy
    output_dir = args.output_dir

    use_cpu = args.use_cpu
    gpu = args.gpu

    # Setup
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Create output directory
    if output_dir is None:
        output_dir = f"evaluation_results/{wandb_name}"

    if os.path.exists(output_dir):
        if force_eval:
            print(
                f"Warning: Output directory {output_dir} already exists. Overwriting."
            )
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. Use --force_eval to overwrite."
            )
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Generate a checkpoint name for saving results
    checkpoint_name = f"{wandb_name}_{artifact_version}"

    # Check for existing evaluation results
    existing_results = check_existing_evaluation(checkpoint_name, output_dir)

    if existing_results is None:
        # Load normalizer
        coarse_data_path = "./Data/coarse_grf_10k.npy"
        fine_data_path = "./Data/fine_grf_10k.npy"
        coarse_normalizer = DatasetNormalization(coarse_data_path)

        # Load model from wandb artifact
        model = load_model_from_wandb(
            wandb_project, wandb_name, artifact_version, device
        )

        # Create test dataset
        te_data_0 = FieldsDataset(
            data_path=coarse_data_path,
            train=False,
            transform=None,
        )

        # Compute L2 losses over test set
        print(f"\nEvaluating L2 losses for {wandb_name} ({artifact_version})")
        results = compute_mean_l2_loss_over_test_set(
            te_data_0,
            model,
            coarse_normalizer,
            device,
            n_samples,
            n_fields,
            discretization_steps,
            energy,
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
        print(f"\nUsing existing evaluation results for: {checkpoint_name}")

    # Print results
    print(
        f"Overall L2 loss: {results_dict['overall_mean']:.4f} ± {results_dict['overall_std']:.4f}\n"
        f"Overall Relative L2 loss: {results_dict['overall_relative_mean']:.4f} ± {results_dict['overall_relative_std']:.4f}"
    )

    # Generate visualization samples
    coarse_normalizer = DatasetNormalization(coarse_data_path)
    fine_normalizer = DatasetNormalization(fine_data_path)

    # Load model again (or reuse if still in memory)
    if "model" not in locals():
        model = load_model_from_wandb(
            wandb_project, wandb_name, artifact_version, device
        )

    # Load test datasets
    te_data_0 = FieldsDataset(data_path=coarse_data_path, train=False, transform=None)
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
    s_path = th.zeros((discretization_steps + 1, batch_size, 1, 32, 32), device=device)
    p_path = th.zeros((discretization_steps, batch_size, 1, 32, 32), device=device)

    s_path[0] = coarse_field_batch

    with th.no_grad():
        euler_discretization(
            x=s_path,
            nn=model,  # Use the loaded model variable
            energy=sigma,
            chunk_size=batch_size,
            store_control=False,
            direction_fn="fwd",
        )

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
    parser = argparse.ArgumentParser(
        description="Evaluate a BM2 forward generation model"
    )

    # WandB and model options
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="jyyresearch/BM2_GRF_MultiscaleBridge",
        help="WandB project path",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        required=True,
        help="WandB run name (e.g., 'mango-mousse-158')",
    )
    parser.add_argument(
        "--artifact_version",
        type=str,
        default="latest",
        help="Artifact version (e.g., 'latest', 'v0', 'v9')",
    )

    # Data paths
    parser.add_argument(
        "--coarse_data_path",
        type=str,
        default="./Data/coarse_grf_10k.npy",
        help="Path to coarse data file",
    )
    parser.add_argument(
        "--fine_data_path",
        type=str,
        default="./Data/fine_grf_10k.npy",
        help="Path to fine data file",
    )

    # Evaluation options
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for generation"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples for computing L2 loss",
    )
    parser.add_argument(
        "--n_fields", type=int, default=500, help="Number of fields to evaluate"
    )
    parser.add_argument(
        "--discretization_steps",
        type=int,
        default=30,
        help="Number of discretization steps",
    )
    parser.add_argument(
        "--energy", type=float, default=1.0, help="Energy parameter for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: 'evaluation_results/{wandb_name}')",
    )
    parser.add_argument(
        "--force_eval",
        action="store_true",
        help="Force re-evaluation even if results already exist",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"evaluation_results/{args.wandb_name}"

    main(args)
