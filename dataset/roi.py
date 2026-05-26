"""ROI (Region of Interest) computation for DIC datasets.

Computes the valid overlap region between reference and target images
after deformation, defined as: {x | x + u(x) is within image bounds}.
"""
import numpy as np


def compute_valid_mask(u_field: np.ndarray) -> np.ndarray:
    """Compute binary mask of valid pixels in the reference frame.

    A pixel at position (x, y) in the reference image is considered valid
    if its displaced position (x + u_x, y + u_y) falls within the image
    bounds. Pixels outside the mask are warped out-of-bounds in the target
    image and should be excluded from training/evaluation.

    Args:
        u_field: [H, W, 2] displacement field (pixel units)

    Returns:
        valid_mask: [H, W] bool, True where the pixel is valid
    """
    H, W = u_field.shape[:2]
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    dst_x = x_coords + u_field[..., 0]
    dst_y = y_coords + u_field[..., 1]

    valid = (dst_x >= 0) & (dst_x < W) & (dst_y >= 0) & (dst_y < H)
    return valid
