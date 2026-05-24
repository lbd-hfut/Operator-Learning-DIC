"""Data augmentation for DIC image pairs.

All augmentations must apply the same geometric transform to both images
and consistently transform the displacement field.
"""
import numpy as np
from typing import Tuple


class DICAugmentation:
    """Augmentation transforms for DIC training.

    Ensures geometric consistency: when flipping/rotating images,
    the displacement field is transformed correspondingly.

    Args:
        image_size: (H, W) image dimensions
        h_flip_prob: probability of horizontal flip
        v_flip_prob: probability of vertical flip
        brightness_range: (min, max) multiplier for brightness jitter
        contrast_range: (min, max) multiplier for contrast jitter
        seed: random seed
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        h_flip_prob: float = 0.5,
        v_flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        seed: int = None,
    ):
        self.image_size = image_size
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        ref_img: np.ndarray,
        tar_img: np.ndarray,
        u_field: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentations.

        Args:
            ref_img: [H, W] reference image
            tar_img: [H, W] target image
            u_field: [H, W, 2] displacement field

        Returns:
            augmented (ref_img, tar_img, u_field)
        """
        ref, tar, u = ref_img.copy(), tar_img.copy(), u_field.copy()

        # Horizontal flip
        if self.rng.random() < self.h_flip_prob:
            ref = np.fliplr(ref)
            tar = np.fliplr(tar)
            u = np.fliplr(u)
            u[..., 0] = -u[..., 0]

        # Vertical flip
        if self.rng.random() < self.v_flip_prob:
            ref = np.flipud(ref)
            tar = np.flipud(tar)
            u = np.flipud(u)
            u[..., 1] = -u[..., 1]

        # Brightness/contrast jitter (applied identically to both images)
        brightness = self.rng.uniform(*self.brightness_range)
        contrast = self.rng.uniform(*self.contrast_range)
        for img in [ref, tar]:
            img *= contrast
            img += (brightness - 1.0) * 0.5
            np.clip(img, 0.0, 1.0, out=img)

        return ref.astype(np.float32), tar.astype(np.float32), u.astype(np.float32)
