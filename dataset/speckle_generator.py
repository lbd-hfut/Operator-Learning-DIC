"""Speckle pattern generation.

Generates synthetic speckle patterns by summing 2D Gaussian blobs with
random positions, sizes, and intensities, mimicking real DIC speckle patterns.
"""
import numpy as np


class SpeckleGenerator:
    """Generate synthetic speckle images for DIC.

    Uses a sum-of-Gaussians model: each speckle particle is a 2D Gaussian
    placed at a random position with random size and intensity.

    Args:
        image_size: (H, W) output image dimensions
        particle_size_range: (min, max) radius of Gaussian particles in pixels
        density: fraction of image area covered by particles [0, 1]
        contrast: intensity range of particles relative to background
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        image_size: tuple = (256, 256),
        particle_size_range: tuple = (1.5, 6.0),
        density: float = 0.4,
        contrast: float = 0.6,
        seed: int = None,
    ):
        self.image_size = image_size
        self.particle_size_range = particle_size_range
        self.density = density
        self.contrast = contrast
        self.rng = np.random.RandomState(seed)

    def generate(self) -> np.ndarray:
        """Generate a single speckle pattern.

        Returns:
            np.ndarray [H, W] float32, values in [0, 1]
        """
        H, W = self.image_size
        area = H * W

        # Estimate number of particles needed to achieve target density
        avg_radius = (self.particle_size_range[0] + self.particle_size_range[1]) / 2
        avg_area = np.pi * avg_radius**2
        n_particles = max(50, int(self.density * area / avg_area))

        # Initialize blank image
        image = np.zeros((H, W), dtype=np.float32)
        y_grid, x_grid = np.mgrid[0:H, 0:W]

        for _ in range(n_particles):
            cx = self.rng.uniform(0, W)
            cy = self.rng.uniform(0, H)
            sigma = self.rng.uniform(
                self.particle_size_range[0] / 2, self.particle_size_range[1] / 2
            )
            intensity = 0.5 + self.rng.uniform(-self.contrast / 2, self.contrast / 2)

            gaussian = np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma**2)
            )
            image += intensity * gaussian

        # Clip to valid range
        image = np.clip(image, 0.0, 1.0)
        return image.astype(np.float32)

    def generate_batch(self, n: int) -> np.ndarray:
        """Generate n speckle patterns.

        Returns:
            np.ndarray [n, H, W] float32
        """
        return np.stack([self.generate() for _ in range(n)])
