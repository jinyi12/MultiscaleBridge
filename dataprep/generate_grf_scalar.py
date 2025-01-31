import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy import stats


class RandomFieldGenerator2D:
    def __init__(self, nx=100, ny=100, lx=1.0, ly=1.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly

        self.x = np.linspace(0, lx, nx)
        self.y = np.linspace(0, ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.points = np.column_stack((self.X.ravel(), self.Y.ravel()))

    def exponential_covariance(self, h, correlation_length):
        return np.exp(-h / correlation_length)

    def gaussian_covariance(self, h, correlation_length):
        return np.exp(-((h / correlation_length) ** 2))

    def generate_random_field(
        self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"
    ):
        n_points = len(self.points)
        h = cdist(self.points, self.points)
        if covariance_type == "exponential":
            C = self.exponential_covariance(h, correlation_length)
        else:
            C = self.gaussian_covariance(h, correlation_length)
        C += 1e-8 * np.eye(n_points)  # nugget
        L = np.linalg.cholesky(C)
        Z = np.random.normal(0, 1, n_points)
        random_values = mean + std * (L @ Z)
        return random_values.reshape(self.nx, self.ny)

    def coarsen_field(
        self, field, filter_sigma=2.0, downsample_factor=2, method="linear"
    ):
        """
        Coarsen a field by smoothing and downsampling

        Args:
            field: 2D numpy array
            filter_sigma: Gaussian filter width
            downsample_factor: Integer factor to reduce resolution by
            method: 'linear' or 'cubic' interpolation
        """
        # Input validation
        if not isinstance(field, np.ndarray) or field.ndim != 2:
            raise ValueError("Field must be a 2D numpy array")

        # First apply Gaussian smoothing
        smooth = gaussian_filter(field, sigma=filter_sigma)

        # Create coordinate grids for original field
        ny, nx = smooth.shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)

        # Create interpolator
        f = RegularGridInterpolator((y, x), smooth, method=method, bounds_error=False)

        # Create coarse grid points
        nx_coarse = nx // downsample_factor
        ny_coarse = ny // downsample_factor
        x_coarse = np.linspace(0, 1, nx_coarse)
        y_coarse = np.linspace(0, 1, ny_coarse)

        # Create mesh of query points
        X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
        query_points = np.stack((Y_coarse.ravel(), X_coarse.ravel()), axis=-1)

        # Interpolate and reshape
        coarse = f(query_points).reshape(ny_coarse, nx_coarse)

        return coarse

    def compute_statistics(self, field):
        flat_field = field.ravel()
        return {
            "mean": np.mean(flat_field),
            "std": np.std(flat_field),
            "skewness": stats.skew(flat_field),
            "kurtosis": stats.kurtosis(flat_field),
            "min": np.min(flat_field),
            "max": np.max(flat_field),
            "median": np.median(flat_field),
        }

    def compute_correlation_length(self, field):
        mean_field = np.mean(field)
        norm_field = field - mean_field
        acf_x = np.zeros(self.nx // 2)

        # One fix is to make acf_x match the maximum needed index and loop only up to shape[1] - 1:
        n = norm_field.shape[1]
        acf_x = np.zeros(n, dtype=float)

        # Optionally handle zero-lag separately:
        acf_x[0] = np.mean(norm_field * norm_field)

        # Now safely loop from 1 .. n - 1
        for i in range(1, n):
            acf_x[i] = np.mean(norm_field[:, :-i] * norm_field[:, i:])

        acf_x /= acf_x[0]
        idx = np.argmin(np.abs(acf_x - 1 / np.e))
        acf_x /= acf_x[0]

        idx = np.argmin(np.abs(acf_x - 1 / np.e))
        correlation_length = self.x[idx]
        return correlation_length

    def plot_fields_and_stats(self, original_field, coarse_field):
        """Plot original and coarse fields with statistics"""
        fig = plt.figure(figsize=(15, 8))

        # Field visualizations (top row)
        fields = [original_field, coarse_field]
        titles = ["Original Field", "Coarse Field"]

        for i, (field, title) in enumerate(zip(fields, titles)):
            ax = plt.subplot(2, 3, i + 1)
            im = ax.imshow(field, extent=[0, self.lx, 0, self.ly])
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

        # Histogram comparison
        ax_hist = plt.subplot(2, 3, 3)
        ax_hist.hist(original_field.ravel(), bins=50, alpha=0.5, label="Original")
        ax_hist.hist(coarse_field.ravel(), bins=50, alpha=0.5, label="Coarse")
        ax_hist.set_title("Histogram Comparison")
        ax_hist.legend()

        # X-direction profiles
        ax_profile = plt.subplot(2, 3, 4)
        mid_y = self.ny // 2
        mid_y_coarse = coarse_field.shape[0] // 2
        x_orig = np.linspace(0, self.lx, original_field.shape[1])
        x_coarse = np.linspace(0, self.lx, coarse_field.shape[1])

        print("X_coarse shape:", x_coarse.shape)
        print("Coarse field shape:", coarse_field.shape)

        ax_profile.plot(
            x_orig, original_field[mid_y, :], "b-", label="Original", alpha=0.7
        )
        ax_profile.plot(
            x_coarse, coarse_field[mid_y_coarse, :], "r--", label="Coarse", alpha=0.7
        )
        ax_profile.set_title("X-direction Profile at Mid-Y")
        ax_profile.legend()

        # Statistics table
        ax_stats = plt.subplot(2, 3, 5)
        stats_list = [self.compute_statistics(f) for f in fields]
        rows = ["Mean", "Std", "Skewness", "Kurtosis", "Min", "Max", "Median"]
        cell_text = []
        for row in rows:
            cell_text.append([f"{stats[row.lower()]:.3f}" for stats in stats_list])

        ax_stats.table(
            cellText=cell_text,
            rowLabels=rows,
            colLabels=["Original", "Coarse"],
            loc="center",
        )
        ax_stats.axis("off")
        ax_stats.set_title("Statistical Properties")

        # Correlation analysis
        ax_corr = plt.subplot(2, 3, 6)
        # Create interpolator for coarse field
        y_coarse = x_coarse  # Assuming square domain
        interp = RegularGridInterpolator(
            (y_coarse, x_coarse), coarse_field, method="linear", bounds_error=False
        )

        # Create query points
        X_orig, Y_orig = np.meshgrid(x_orig, x_orig)
        query_points = np.stack((Y_orig.ravel(), X_orig.ravel()), axis=-1)

        # Interpolate coarse field to original grid
        coarse_interp = interp(query_points).reshape(original_field.shape)
        ax_corr.scatter(original_field.ravel(), coarse_interp.ravel(), alpha=0.1)
        ax_corr.set_title("Original vs Coarse")

        plt.tight_layout()
        return fig


import numpy as np
import os

os.makedirs("./Data", exist_ok=True)

# After defining RandomFieldGenerator2D as above:
generator_fine = RandomFieldGenerator2D(nx=32, ny=32)  # for a fine grid
generator_coarse = RandomFieldGenerator2D(nx=16, ny=16)  # for a coarser grid

n_samples = 1000
fine_list = []
coarse_list = []

correlation_length = 0.2
spatial_lag = generator_fine.lx / generator_fine.nx
ratio = spatial_lag / correlation_length
print(
    f"Spatial lag: {spatial_lag:.3f}, Correlation length: {correlation_length:.3f}, Ratio: {ratio:.3f}"
)

for _ in range(n_samples):
    fine_field = generator_fine.generate_random_field(
        mean=10.0,
        std=2.0,
        correlation_length=correlation_length,
        covariance_type="exponential",
    )
    coarse_field = generator_coarse.coarsen_field(fine_field, filter_sigma=2.0)
    fine_list.append(fine_field.reshape(generator_fine.ny, generator_fine.nx))
    coarse_list.append(coarse_field.reshape(generator_coarse.ny, generator_coarse.nx))

np.save("./Data/fine_grf_1k.npy", np.stack(fine_list))
np.save("./Data/coarse_grf_1k.npy", np.stack(coarse_list))
