"""Image warping using displacement fields (forward splatting).

Forward warp: for each reference pixel at (x, y), push its value to
target position (x + u_x, y + u_y). Sub-pixel positions are handled
by bilinear splatting — the value is distributed to the 4 nearest
target pixels with bilinear weights.

This ensures u_field is read at the correct (reference) coordinate,
unlike inverse warping which reads u at target coordinates.
"""
import numpy as np


def warp_image(
    image: np.ndarray,
    u_field: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Forward warp via bilinear splatting.

    For each reference pixel at (x, y) with value I, push I to the
    target pixel(s) surrounding (x + u_x, y + u_y).

    Args:
        image: [H, W] reference image, float32
        u_field: [H, W, 2] displacement in pixels, u[...,0]=u_x, u[...,1]=u_y
        order: kept for API compatibility (ignored, always bilinear)
        mode: boundary handling ('constant' or 'nearest')
        cval: fill value for pixels with no reference contribution

    Returns:
        [H, W] warped (target) image, float32
    """
    H, W = image.shape

    # Target position of each reference pixel
    dst_x = np.arange(W, dtype=np.float32) + u_field[..., 0]  # [H, W]
    dst_y = np.arange(H, dtype=np.float32)[:, None] + u_field[..., 1]  # [H, W]

    # Integer and fractional parts
    x0 = np.floor(dst_x).astype(np.int32)
    y0 = np.floor(dst_y).astype(np.int32)
    dx = dst_x - x0.astype(np.float32)
    dy = dst_y - y0.astype(np.float32)

    # Bilinear weights
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    # Allocate target and weight accumulators
    target = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    # Filter to pixels whose target position is within bounds
    valid = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
    vy, vx = np.where(valid)

    img_vals = image[vy, vx]
    _x0 = x0[vy, vx]
    _y0 = y0[vy, vx]
    _dx = dx[vy, vx]
    _dy = dy[vy, vx]
    _w00 = w00[vy, vx]
    _w10 = w10[vy, vx]
    _w01 = w01[vy, vx]
    _w11 = w11[vy, vx]

    # Helper: accumulate with bounds check for x0+1 / y0+1
    def _splat(ty, tx, vals, w):
        m = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
        np.add.at(target, (ty[m], tx[m]), vals[m] * w[m])
        np.add.at(weight, (ty[m], tx[m]), w[m])

    _splat(_y0, _x0, img_vals, _w00)
    _splat(_y0, _x0 + 1, img_vals, _w10)
    _splat(_y0 + 1, _x0, img_vals, _w01)
    _splat(_y0 + 1, _x0 + 1, img_vals, _w11)

    # Normalize: divide accumulated value by accumulated weight
    mask = weight > 0
    result = np.full((H, W), cval, dtype=np.float32)
    result[mask] = (target[mask] / weight[mask]).astype(np.float32)

    if mode == "nearest":
        # Fill holes with nearest-neighbor (simple inpainting)
        from scipy.ndimage import distance_transform_edt
        if not mask.all():
            # Distance transform from valid pixels, fill each hole with
            # the value of the nearest valid pixel
            dist, idx = distance_transform_edt(
                ~mask, return_distances=True, return_indices=True
            )
            result = result[idx[0], idx[1]]

    return result.astype(np.float32)


def warp_image_torch(
    image: "torch.Tensor",
    u_field: "torch.Tensor",
    mode: str = "bilinear",
    padding_mode: str = "zeros",
):
    """PyTorch-based inverse warping using grid_sample.

    Uses inverse warp (grid_sample) which is the native PyTorch primitive.
    For forward warp, use the numpy version above.

    Args:
        image: [B, 1, H, W] reference image
        u_field: [B, H, W, 2] displacement field in pixels
        mode: interpolation mode ('bilinear', 'bicubic')
        padding_mode: boundary handling ('zeros', 'border', 'reflection')

    Returns:
        [B, 1, H, W] warped image
    """
    import torch
    import torch.nn.functional as F

    B, _, H, W = image.shape

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing="ij",
    )
    grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    u_norm = u_field.clone()
    u_norm[..., 0] = u_field[..., 0] * 2.0 / W
    u_norm[..., 1] = u_field[..., 1] * 2.0 / H

    sampling_grid = grid - u_norm

    warped = F.grid_sample(
        image, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )
    return warped
