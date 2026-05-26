"""Deformation field generation.

Generates synthetic displacement fields using analytical functions
covering common deformation modes: tension, compression, shear, rotation,
and composite combinations.
"""
import numpy as np
from typing import Tuple, List


class DeformationGenerator:
    """Generate analytical displacement fields for DIC training.

    Each mode produces a full-field displacement u(x, y) = (u_x, u_y)
    across the entire image domain.

    Args:
        image_size: (H, W) output field dimensions
        displacement_range: (min, max) displacement magnitude in pixels
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        image_size: tuple = (256, 256),
        displacement_range: tuple = (0.1, 20.0),
        seed: int = None,
    ):
        self.image_size = image_size
        self.displacement_range = displacement_range
        self.rng = np.random.RandomState(seed)

    def _normalize_coords(self):
        """Create normalized coordinate grids x, y in [0, 1]."""
        H, W = self.image_size
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        return np.meshgrid(x, y)  # shape: [H, W] each

    def generate(self, mode: str = None, amplitude: float = None) -> np.ndarray:
        """Generate a displacement field.

        Args:
            mode: deformation type. If None, randomly selects from available modes.
            amplitude: peak displacement in pixels. If None, random within range.

        Returns:
            np.ndarray [H, W, 2] displacement field (pixel units)
        """
        if mode is None:
            modes = [
                "tension", "compression", "shear", "rotation", "composite",
                "multiscale_random",
            ]
            mode = self.rng.choice(modes)

        if amplitude is None:
            if mode == "multiscale_random":
                amplitude = self.rng.uniform(0.3, 1.0)
            else:
                amplitude = self.rng.uniform(*self.displacement_range)

        method = getattr(self, f"_gen_{mode}")
        return method(amplitude)

    def _gen_tension(self, amplitude: float) -> np.ndarray:
        """Uniaxial tension along x."""
        x, y = self._normalize_coords()
        u_x = amplitude * (x - 0.5)  # stretch from center
        u_y = -0.3 * amplitude * (y - 0.5)  # Poisson contraction
        return np.stack([u_x, u_y], axis=-1).astype(np.float32)

    def _gen_compression(self, amplitude: float) -> np.ndarray:
        """Uniaxial compression along x."""
        u = self._gen_tension(amplitude)
        return -u

    def _gen_shear(self, amplitude: float) -> np.ndarray:
        """Simple shear: u_x proportional to y."""
        x, y = self._normalize_coords()
        u_x = amplitude * (y - 0.5)
        u_y = np.zeros_like(x)
        return np.stack([u_x, u_y], axis=-1).astype(np.float32)

    def _gen_rotation(self, amplitude: float) -> np.ndarray:
        """Rigid body rotation around image center."""
        x, y = self._normalize_coords()
        angle = amplitude * 0.05  # convert amplitude to rotation angle
        cx, cy = 0.5, 0.5
        dx, dy = x - cx, y - cy
        u_x = np.cos(angle) * dx - np.sin(angle) * dy - dx
        u_y = np.sin(angle) * dx + np.cos(angle) * dy - dy
        return np.stack([u_x, u_y], axis=-1).astype(np.float32)

    def _gen_composite(self, amplitude: float) -> np.ndarray:
        """Composite deformation: tension + shear + some nonlinearity."""
        x, y = self._normalize_coords()
        u_x = amplitude * (
            0.5 * (x - 0.5)
            + 0.3 * (y - 0.5)
            + 0.1 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        )
        u_y = amplitude * (
            0.3 * (y - 0.5)
            + 0.1 * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x)
        )
        return np.stack([u_x, u_y], axis=-1).astype(np.float32)

    def _gen_multiscale_random(self, amplitude: float = None) -> np.ndarray:
        """Multi-scale random deformation field (sub-pixel, < 1 px).

        From "When Deep Learning Meets Digital Image Correlation":
        Random displacements at control points on a coarse grid, bicubic
        interpolated to full resolution. Boundaries are zeroed. The region
        size is randomly chosen per sample from [128, 64, 32, 16, 8, 4].

        This produces complex, non-smooth fields suitable for testing
        sub-pixel DIC accuracy, where traditional IC-GN methods struggle.

        Args:
            amplitude: max displacement in pixels. If None, defaults to ~1.0.

        Returns:
            np.ndarray [H, W, 2] displacement field (pixel units)
        """
        from scipy.ndimage import zoom

        if amplitude is None:
            amplitude = 1.0

        H, W = self.image_size

        # Randomly select region size
        region_sizes = [128, 64, 32, 16, 8, 4]
        s = region_sizes[self.rng.randint(0, len(region_sizes))]

        grid_h = H // s + 3
        grid_w = W // s + 3

        # Random displacements at control points, bounded to ±amplitude
        f = self.rng.uniform(-amplitude, amplitude, (grid_h, grid_w))
        g = self.rng.uniform(-amplitude, amplitude, (grid_h, grid_w))

        # Bicubic interpolation (order=3) from coarse grid to full resolution
        zoom_h = H / grid_h
        zoom_w = W / grid_w
        u_x = zoom(f, (zoom_h, zoom_w), order=3)
        u_y = zoom(g, (zoom_h, zoom_w), order=3)

        # Zero out boundaries (2 px each side)
        u_x[:2, :] = 0
        u_y[:2, :] = 0
        u_x[-2:, :] = 0
        u_y[-2:, :] = 0
        u_x[:, :2] = 0
        u_y[:, :2] = 0
        u_x[:, -2:] = 0
        u_y[:, -2:] = 0

        return np.stack([u_x, u_y], axis=-1).astype(np.float32)
