"""Route A-specific losses.

Includes grayscale consistency auxiliary loss: |I_ref(x) - I_tar(x + u_pred(x))|.
Differentiable via grid_sample — compares the reference image with the warped target.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GrayLevelConsistencyLoss(nn.Module):
    """Grayscale invariance loss: ||I_ref - I_tar_warped||_1.

    Warps the target image using predicted displacement and compares
    with the reference image. This enforces the physical constraint
    I_ref(x) = I_tar(x + u(x)) directly.

    Args:
        mode: interpolation mode for grid_sample
        padding_mode: boundary handling
    """

    def __init__(self, mode: str = "bilinear", padding_mode: str = "border"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self,
        ref_img: torch.Tensor,
        tar_img: torch.Tensor,
        u_pred: torch.Tensor,
        query_points: torch.Tensor,
        query_mask: torch.Tensor = None,
        image_size: tuple = (256, 256),
    ) -> torch.Tensor:
        """Compute grayscale consistency loss at query points.

        At each query point x, we warp x to x + u_pred(x) and compare:
          L = |I_ref(x) - I_tar(x + u_pred(x))|

        Args:
            ref_img: [B, 1, H, W]
            tar_img: [B, 1, H, W]
            u_pred: [B, N, 2] displacement in pixels at query points
            query_points: [B, N, 2] normalized coordinates
            query_mask: [B, N] valid query point mask
            image_size: (H, W) used for coordinate conversion

        Returns:
            scalar mean absolute error
        """
        B, _, H, W = ref_img.shape

        # Convert normalized query coords to [-1, 1] grid coordinates
        # query_points in [0,1]^2 -> grid in [-1, 1]^2
        grid = query_points * 2.0 - 1.0  # [B, N, 2]

        # Add displacement in normalized coordinates
        u_norm = u_pred.clone()
        u_norm[..., 0] = u_pred[..., 0] * 2.0 / W
        u_norm[..., 1] = u_pred[..., 1] * 2.0 / H

        # For inverse warp: we want I_ref(x) = I_tar(x + u(x))
        # So sample target at position x + u(x)
        sample_grid = grid + u_norm  # [B, N, 2]
        sample_grid = sample_grid.unsqueeze(2)  # [B, N, 1, 2]

        tar_warped = F.grid_sample(
            tar_img, sample_grid, mode=self.mode,
            padding_mode=self.padding_mode, align_corners=True,
        )  # [B, 1, N, 1]

        tar_warped = tar_warped.squeeze(-1).squeeze(1)  # [B, N]

        # Sample reference at query points
        ref_grid = grid.unsqueeze(2)
        ref_sampled = F.grid_sample(
            ref_img, ref_grid, mode=self.mode,
            padding_mode=self.padding_mode, align_corners=True,
        )
        ref_sampled = ref_sampled.squeeze(-1).squeeze(1)  # [B, N]

        diff = torch.abs(ref_sampled - tar_warped)

        if query_mask is not None:
            diff = diff * query_mask.float()
            loss = diff.sum() / query_mask.sum().clamp(min=1)
        else:
            loss = diff.mean()

        return loss
