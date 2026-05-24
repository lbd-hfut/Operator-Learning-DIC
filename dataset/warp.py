"""Image warping using displacement fields.

Applies warp I_tar(x) = I_ref(x - u(x)) via bilinear interpolation,
which is the inverse warping convention suitable for DIC.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def warp_image(
    image: np.ndarray,
    u_field: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Warp image using displacement field via bilinear interpolation.

    I_tar(x) = I_ref(x - u(x))  [inverse warp]

    At each pixel location (i, j) in the target image, we look up the
    source location (i - u_y[i,j], j - u_x[i,j]) in the reference image.
    This avoids holes and ensures the output is densely defined.

    Args:
        image: [H, W] reference image, float32
        u_field: [H, W, 2] displacement field in pixels, u[...,0]=u_x, u[...,1]=u_y
        order: interpolation order (1=bilinear, 3=cubic)
        mode: boundary handling ('constant', 'nearest', 'mirror', 'wrap')
        cval: fill value for 'constant' mode

    Returns:
        [H, W] warped (target) image, float32
    """
    H, W = image.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)

    # Inverse warp: sample from source at (x - u_x, y - u_y)
    src_x = x_coords - u_field[..., 0]
    src_y = y_coords - u_field[..., 1]

    # map_coordinates takes (rows, cols) = (y, x)
    coords = np.stack([src_y.ravel(), src_x.ravel()])

    warped = map_coordinates(image, coords, order=order, mode=mode, cval=cval)
    return warped.reshape(H, W).astype(np.float32)


def warp_image_torch(
    image: "torch.Tensor",
    u_field: "torch.Tensor",
    mode: str = "bilinear",
    padding_mode: str = "zeros",
):
    """PyTorch-based image warping using grid_sample.

    Args:
        image: [B, 1, H, W] reference image
        u_field: [B, H, W, 2] displacement field in pixels, u[...,0]=u_x, u[...,1]=u_y
        mode: interpolation mode ('bilinear', 'bicubic')
        padding_mode: boundary handling ('zeros', 'border', 'reflection')

    Returns:
        [B, 1, H, W] warped image
    """
    import torch
    import torch.nn.functional as F

    B, _, H, W = image.shape

    # Build sampling grid in normalized coordinates [-1, 1]
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing="ij",
    )
    grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Convert pixel displacement to normalized coordinate shift
    u_norm = u_field.clone()
    u_norm[..., 0] = u_field[..., 0] * 2.0 / W
    u_norm[..., 1] = u_field[..., 1] * 2.0 / H

    sampling_grid = grid - u_norm

    warped = F.grid_sample(
        image, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )
    return warped
