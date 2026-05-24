"""Query point sampling strategies for DIC operator training.

Query points are the locations where the operator is asked to predict
displacement. Various sampling strategies trade off training signal coverage
vs. computational efficiency.
"""
import numpy as np
from typing import Tuple


class QueryPointSampler:
    """Sample query points from the image domain.

    Supports uniform, gradient-weighted, and ROI-masked sampling.

    Args:
        image_size: (H, W) image dimensions
        seed: random seed for reproducibility
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 256), seed: int = None):
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        n_points: int,
        u_field: np.ndarray = None,
        roi_mask: np.ndarray = None,
        gradient_weighted: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample query points and extract ground truth displacements.

        Args:
            n_points: number of query points to sample
            u_field: [H, W, 2] full-field displacement (for GT extraction)
            roi_mask: [H, W] bool, True = valid region
            gradient_weighted: if True, oversamples high-gradient regions

        Returns:
            query_points: [N, 2] normalized coordinates in [0, 1]²
            u_gt: [N, 2] displacement values in pixels at query points
        """
        H, W = self.image_size

        if roi_mask is not None and gradient_weighted and u_field is not None:
            pts = self._sample_gradient_weighted(n_points, u_field, roi_mask)
        elif roi_mask is not None:
            pts = self._sample_uniform_masked(n_points, roi_mask)
        else:
            pts = self._sample_uniform(n_points)

        # Extract ground truth at sampled points
        if u_field is not None:
            u_gt = self._sample_field(u_field, pts)
        else:
            u_gt = np.zeros((n_points, 2), dtype=np.float32)

        return pts, u_gt

    def _sample_uniform(self, n: int) -> np.ndarray:
        """Uniform random sampling over entire image."""
        return self.rng.uniform(0, 1, (n, 2)).astype(np.float32)

    def _sample_uniform_masked(self, n: int, mask: np.ndarray) -> np.ndarray:
        """Uniform random sampling within ROI mask."""
        valid_indices = np.argwhere(mask)  # [N_valid, 2] (y, x)
        if len(valid_indices) == 0:
            return self._sample_uniform(n)

        H, W = self.image_size
        idx = self.rng.choice(len(valid_indices), n, replace=True)
        pts_yx = valid_indices[idx].astype(np.float32)
        # Convert to normalized coordinates
        pts = np.zeros((n, 2), dtype=np.float32)
        pts[:, 0] = pts_yx[:, 1] / (W - 1)  # x
        pts[:, 1] = pts_yx[:, 0] / (H - 1)  # y
        return pts

    def _sample_gradient_weighted(
        self, n: int, u_field: np.ndarray, mask: np.ndarray = None
    ) -> np.ndarray:
        """Sample with probability proportional to displacement gradient magnitude."""
        H, W = self.image_size
        gy, gx = np.gradient(u_field[..., 0]), np.gradient(u_field[..., 1])
        grad_mag = np.sqrt(gy[0] ** 2 + gy[1] ** 2 + gx[0] ** 2 + gx[1] ** 2)

        if mask is not None:
            grad_mag = grad_mag * mask

        grad_mag += 1e-6
        probs = grad_mag.ravel() / grad_mag.sum()

        idx = self.rng.choice(H * W, n, p=probs, replace=True)
        y, x = np.unravel_index(idx, (H, W))

        pts = np.stack([x / (W - 1), y / (H - 1)], axis=-1).astype(np.float32)
        return pts

    def _sample_field(self, u_field: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Extract displacement values at query point locations via bilinear interpolation."""
        H, W = self.image_size[:2]
        px = pts[:, 0] * (W - 1)
        py = pts[:, 1] * (H - 1)

        # Bilinear interpolation
        x0 = np.floor(px).astype(int)
        y0 = np.floor(py).astype(int)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)

        dx = px - x0
        dy = py - y0

        # Gather values at 4 corners
        v00 = u_field[y0, x0]  # [N, 2]
        v10 = u_field[y0, x1]
        v01 = u_field[y1, x0]
        v11 = u_field[y1, x1]

        dx = dx[:, None]
        dy = dy[:, None]

        u = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
        return u.astype(np.float32)
