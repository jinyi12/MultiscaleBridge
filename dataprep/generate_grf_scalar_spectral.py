import numpy as np
import os
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy import stats
import argparse

import torch as th
import torchvision.transforms.functional as F


class RandomFieldGenerator2D:
    def __init__(self, nx=100, ny=100, lx=1.0, ly=1.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly

        self.x = np.linspace(0, lx, nx)
        self.y = np.linspace(0, ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.points = np.column_stack((self.X.ravel(), self.Y.ravel()))

    def generate_random_field_cholesky(
        self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"
    ):
        """Original method using Cholesky decomposition for comparison."""
        n_points = len(self.points)
        h = cdist(self.points, self.points)
        if covariance_type == "exponential":
            C = np.exp(-h / correlation_length)
        else:
            C = np.exp(-((h / correlation_length) ** 2))
        C += 1e-8 * np.eye(n_points)  # nugget
        L = np.linalg.cholesky(C)
        Z = np.random.normal(0, 1, n_points)
        random_values = mean + std * (L @ Z)
        return random_values.reshape(self.nx, self.ny)

    def generate_random_field(
        self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"
    ):
        """Spectral method using FFT for efficient generation."""
        dx = self.lx / (self.nx - 1)
        dy = self.ly / (self.ny - 1)

        # Generate white noise and compute FFT
        white_noise = np.random.normal(0, 1, (self.nx, self.ny))
        fourier_coefficients = np.fft.fft2(white_noise)

        # Compute wavevectors
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(Kx**2 + Ky**2)

        # Compute power spectrum
        l = correlation_length
        if covariance_type == "exponential":
            P = (2 * np.pi * l**2) / (1 + (l * K) ** 2) ** (1.5)
        elif covariance_type == "gaussian":
            P = np.pi * l**2 * np.exp(-((l * K) ** 2) / 4)
        else:
            raise ValueError("Invalid covariance_type")
        P = np.nan_to_num(P)

        # Scale Fourier coefficients and inverse FFT
        fourier_coefficients *= np.sqrt(P)
        field = np.fft.ifft2(fourier_coefficients).real

        # Normalize to desired mean and std
        field = (field - np.mean(field)) / np.std(field) * std + mean
        return field

    def coarsen_field(
        self,
        field,
        H,
        downsample_factor=None,
        size=None,
        method="bicubic",
        apply_gaussian_smoothing=True,
        antialias=True,
    ):
        """
        Coarsen a field by smoothing and downsampling.

        Args:
            field (torch.Tensor): Input tensor of shape [B, C, H, W] or [C, H, W].
            H (int): constant * (maximum diameter of inclusion), in this case maximum diameter is just determined from the correlation length as L_c >= 4 * h_e where h_e is the effective element size, h_e = 1 / n_x. Therefore H = 4 in this case for L_c >= 4 * h_e. for greater multiple of h_e, H can be increased, e.g. H = 8 for L_c >= 8 * h_e.
            downsample_factor (int): Factor by which to downsample spatial dimensions.
            method (str): Interpolation method - 'bilinear' or 'bicubic'.

        Returns:
            torch.Tensor: Coarsened tensor.
        """

        # Handle input dimensions
        if isinstance(field, np.ndarray):
            field = th.from_numpy(field)

        if field.dim() == 2:
            # that means it is a single 2D field
            field = field.unsqueeze(0)  # Add batch dimension: [1, H, W]
            field = field.unsqueeze(0)  # Add channel dimension: [1, 1, H, W]
            squeeze = True
        elif field.dim() == 3:
            # that means it is a single 3D field
            field = field.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
            squeeze = True
        elif field.dim() == 4:
            squeeze = False
        else:
            raise ValueError(
                "Field must have 3 or 4 dimensions [C, H, W] or [B, C, H, W]"
            )

        # Calculate kernel size, if H is even, we need to make it odd
        kernel_size = int(H / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1

        filter_sigma = H / 6

        if apply_gaussian_smoothing:
            # Apply Gaussian smoothing
            smooth = F.gaussian_blur(
                field,
                kernel_size=kernel_size,
                sigma=filter_sigma,
            )
        else:
            smooth = field

        # Downsample using interpolation
        if downsample_factor is not None:
            scale_factor = 1.0 / downsample_factor
            coarse = th.nn.functional.interpolate(
                smooth,
                scale_factor=scale_factor,
                mode=method,
                align_corners=False if method != "nearest" else None,
                antialias=antialias,
            )
        elif size is not None:
            coarse = th.nn.functional.interpolate(
                smooth,
                size=size,
                mode=method,
                align_corners=False if method != "nearest" else None,
                antialias=antialias,
            )
        else:
            coarse = smooth

        if squeeze:
            coarse = coarse.squeeze(0)  # Remove batch dimension if it was added

        return coarse

    def compute_correlation_length(self, field, plot_fit=False):
        """
        Compute the two-point correlation function from a 2D field,
        extract its radial profile, and fit the profile to the model:
            model(r; A, ξ, k) = A * exp(-r/ξ) * (r/ξ)^k
        This method returns the fitted correlation length ξ and exponent k.

        Parameters:
            field (numpy.ndarray): The 2D field.
            plot_fit (bool): If True, plot the radial correlation and its fit.

        Returns:
            tuple: (ξ, k) where ξ is the correlation length and k is the exponent.
        """
        # Subtract the mean to obtain a zero-mean field.
        field_zero = field - np.mean(field)

        # Compute the autocorrelation using the Fourier transform method.
        # This implements the Wiener–Khinchin theorem.
        f_fft = np.fft.fft2(field_zero)
        power_spectrum = np.abs(f_fft) ** 2
        corr = np.fft.ifft2(power_spectrum).real
        corr = np.fft.fftshift(corr)

        # Normalize by the zero-lag value (should be at the center of the correlation map).
        center = (corr.shape[0] // 2, corr.shape[1] // 2)
        norm_corr = corr / corr[center]

        # Create a grid of radial distances in physical units.
        nx, ny = field.shape
        dx = self.lx / (nx - 1)
        dy = self.ly / (ny - 1)
        x = (np.arange(nx) - nx // 2) * dx
        y = (np.arange(ny) - ny // 2) * dy
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)

        # Obtain a radial profile by averaging over annuli.
        from scipy.stats import binned_statistic

        nbins = 100
        bin_means, bin_edges, _ = binned_statistic(
            R.ravel(), norm_corr.ravel(), statistic="mean", bins=nbins
        )
        r_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Exclude bins with insufficient data or singular behavior at r = 0.
        # Filter out bins that produce NaNs (or Infs) in the averaged correlation.
        mask = (r_vals > dx) & np.isfinite(bin_means)
        r_fit = r_vals[mask]
        corr_fit = bin_means[mask]

        # Define the fitting model:
        # model(r; A, ξ, k) = A * exp(-r/ξ) * (r/ξ)^k
        def model(r, A, xi, k):
            return A * np.exp(-r / xi) * (r / xi) ** k

        # Use non-linear least squares to fit the radial correlation data.
        # Initial guesses: A ~1 (normalized), ξ ~ self.lx/5, k ~ 0.0.
        from scipy.optimize import curve_fit

        p0 = [1.0, self.lx / 5, 0.0]
        popt, pcov = curve_fit(model, r_fit, corr_fit, p0=p0, maxfev=200000)

        # Optionally, plot the radial correlation data and the fitted model.
        if plot_fit:
            plt.figure()
            plt.scatter(r_vals, bin_means, label="Radial ACF", color="blue")
            r_model = np.linspace(r_fit.min(), r_fit.max(), 200)
            plt.plot(
                r_model,
                model(r_model, *popt),
                "r-",
                label=f"Fit: A={popt[0]:.2f}, ξ={popt[1]:.2f}, k={popt[2]:.2f}",
            )
            plt.xlabel("r")
            plt.ylabel("Normalized Correlation")
            plt.title("Radial Correlation Function and Fitted Model")
            plt.legend()
            plt.show()

        # Return the estimated correlation length (ξ) and exponent (k).
        return popt[1], popt[2]

    def plot_comparison(self, field1, field2, title1="Original", title2="Spectral"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, field, title in zip(axes[0], [field1, field2], [title1, title2]):
            im = ax.imshow(field, extent=[0, self.lx, 0, self.ly])
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        axes[1, 0].hist(field1.ravel(), bins=50, alpha=0.5, label=title1)
        axes[1, 0].hist(field2.ravel(), bins=50, alpha=0.5, label=title2)
        axes[1, 0].legend()
        axes[1, 0].set_title("Histogram Comparison")
        mid = self.nx // 2
        axes[1, 1].plot(self.x, field1[mid, :], label=title1)
        axes[1, 1].plot(self.x, field2[mid, :], label=title2)
        axes[1, 1].legend()
        axes[1, 1].set_title("Mid-Y Profile")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def generate_random_field_kl(
        self,
        mean=10.0,
        std=2.0,
        correlation_length=0.2,
        kernel="exponential",
        truncation_threshold=None,
    ):
        """
        Generate a Gaussian random field using the Karhunen–Loève expansion.

        Parameters:
            mean (float): Mean value of the field.
            std (float): Standard deviation of the field.
            correlation_length (float): Correlation length parameter.
            kernel (str): Type of covariance kernel to use. Options are:
                          "exponential"    -> C(h) = exp(-h / correlation_length)
                          "squared" or "squared_exponential" -> C(h) = exp(-0.5 * (h/correlation_length)**2)
            truncation_threshold (float, optional): If provided, eigenvalues below this threshold will be truncated.

        Returns:
            numpy.ndarray: A 2D field of shape (self.nx, self.ny).
        """
        from scipy.spatial.distance import cdist  # ensure cdist is imported

        n_points = len(self.points)
        # Compute the pairwise Euclidean distances between grid points
        h = cdist(self.points, self.points)

        # Build the covariance matrix depending on the selected kernel
        if kernel == "exponential":
            C = np.exp(-h / correlation_length)
        elif kernel in ["squared", "squared_exponential"]:
            C = np.exp(-0.5 * (h / correlation_length) ** 2)
        else:
            raise ValueError("Invalid kernel type. Use 'exponential' or 'squared'.")

        # Add a small nugget for numerical stability
        C += 1e-8 * np.eye(n_points)

        # Compute eigen-decomposition of the covariance matrix
        eig_val, eig_vec = np.linalg.eigh(C)

        # Sort eigenvalues (and corresponding eigenvectors) in descending order
        idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        # Optionally truncate the KL expansion
        if truncation_threshold is not None:
            keep = eig_val > truncation_threshold
            eig_val = eig_val[keep]
            eig_vec = eig_vec[:, keep]

        # Generate independent standard normal random variables for the KL expansion
        xi = np.random.normal(0, 1, size=eig_val.shape[0])

        # Construct the field using the KL expansion: sum_i sqrt(lambda_i)*xi_i*psi_i
        field_flat = mean + std * (eig_vec @ (np.sqrt(eig_val) * xi))

        # Reshape the flat field into the grid dimensions
        field = field_flat.reshape(self.nx, self.ny)
        return field


def main(args=None):
    """
    Main function to generate Gaussian random fields with specified parameters.

    Parameters:
        args: Command line arguments (if None, they will be parsed from sys.argv)
    """
    parser = argparse.ArgumentParser(description="Generate Gaussian Random Fields")
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--lx", type=float, default=1.0, help="Domain size in x direction"
    )
    parser.add_argument(
        "--ly", type=float, default=1.0, help="Domain size in y direction"
    )
    parser.add_argument("--H", type=int, default=4, help="Size of the field")
    parser.add_argument(
        "--nx_fine", type=int, default=32, help="Number of points in x for fine grid"
    )
    parser.add_argument(
        "--ny_fine", type=int, default=32, help="Number of points in y for fine grid"
    )
    parser.add_argument(
        "--nx_coarse",
        type=int,
        default=16,
        help="Number of points in x for coarse grid",
    )
    parser.add_argument(
        "--ny_coarse",
        type=int,
        default=16,
        help="Number of points in y for coarse grid",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="../Data",
        help="Folder to save generated data",
    )
    parser.add_argument(
        "--mean", type=float, default=10.0, help="Mean of the random field"
    )
    parser.add_argument(
        "--std", type=float, default=2.0, help="Standard deviation of the random field"
    )
    parser.add_argument(
        "--covariance",
        type=str,
        default="exponential",
        choices=["exponential", "gaussian"],
        help="Covariance type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(args)

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)

    # Initialize generators
    generator_fine = RandomFieldGenerator2D(
        nx=args.nx_fine, ny=args.ny_fine, lx=args.lx, ly=args.ly
    )

    corr_length_fine = args.H / args.nx_fine

    # Generate samples
    fine_list = []
    coarse_list = []

    print(
        f"Generating {args.n_samples} samples with correlation length {corr_length_fine}..."
    )
    for i in range(args.n_samples):
        if i % 1000 == 0 and i > 0:
            print(f"Generated {i} samples...")

        # Generate fine field
        fine_field = generator_fine.generate_random_field(
            mean=args.mean,
            std=args.std,
            correlation_length=corr_length_fine,
            covariance_type=args.covariance,
        )

        # Convert to PyTorch tensor and add channel dimension
        fine_tensor = (
            th.from_numpy(fine_field).float().unsqueeze(0)
        )  # Shape becomes [1, H, W]

        # Determine downsample factor based on fine and coarse resolutions
        downsample_factor = args.nx_fine // args.nx_coarse

        # Now coarsen the field
        coarse_tensor = generator_fine.coarsen_field(
            fine_tensor,
            H=args.H,
            downsample_factor=downsample_factor,
            method="bicubic",
            antialias=True,
        )

        # Convert back to numpy for storage
        coarse_field = coarse_tensor.numpy().squeeze()

        fine_list.append(fine_field)
        coarse_list.append(coarse_field)

    # Save the generated fields
    fine_output_path = os.path.join(
        args.output_folder, f"fine_grf_{args.n_samples}_res{args.nx_fine}.npy"
    )
    coarse_output_path = os.path.join(
        args.output_folder, f"coarse_grf_{args.n_samples}_res{args.nx_coarse}.npy"
    )

    np.save(fine_output_path, np.array(fine_list))
    np.save(coarse_output_path, np.array(coarse_list))

    print(f"Saved {args.n_samples} samples to {args.output_folder}")
    print(f"Fine grid shape: {fine_list[0].shape}")
    print(f"Coarse grid shape: {coarse_list[0].shape}")


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
