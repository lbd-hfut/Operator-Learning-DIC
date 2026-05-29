"""Dual-Channel CNN Encoder for Route A.

Encodes the concatenated [I_ref, I_tar] image pair into a feature field F_input.
The first convolutional layer performs local comparison between the two images.
"""
import torch
import torch.nn as nn
import math


class ResBlock(nn.Module):
    """Residual block with two Conv-LayerNorm-GELU layers."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class DualChannelCNNEncoder(nn.Module):
    """CNN encoder for concatenated reference and target images.

    Input:  [B, 2, H, W]  (I_ref and I_tar concatenated channel-wise)
    Output: F_input [B, N, d] where N = H/ds² * W/ds²

    Uses minimal downsampling (stride-2 × ds only) to preserve speckle details.
    """

    def __init__(
        self,
        in_channels: int = 3,  # ref + tar + diff
        channels: tuple = (64, 128, 256),
        kernel_size: int = 7,
        n_blocks: int = 4,
        downsample_factor: int = 2,   # total stride = 2^downsample_factor
        feature_dim: int = 256,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        # Stem: initial convolution with minimal downsampling
        layers = []
        current_ch = in_channels

        # First conv: large kernel, no downsampling (preserve speckle details)
        layers.append(
            nn.Conv2d(current_ch, channels[0], kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        )
        layers.append(nn.GroupNorm(8, channels[0]))
        layers.append(nn.GELU())
        current_ch = channels[0]

        # ResBlock stages with progressive channel increase
        for stage_idx, ch in enumerate(channels):
            stride = 2 if stage_idx < downsample_factor else 1
            for _ in range(n_blocks):
                layers.append(ResBlock(current_ch, ch, stride=stride))
                current_ch = ch
                stride = 1  # Only first block per stage uses stride

        self.cnn = nn.Sequential(*layers)

        # Project to feature dimension
        self.proj = nn.Conv2d(current_ch, feature_dim, 1)

        # Learnable 2D position embeddings on the feature grid
        max_hw = 256  # maximum expected grid size
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_dim, max_hw, max_hw))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, ref_img: torch.Tensor, tar_img: torch.Tensor) -> torch.Tensor:
        """Encode image pair to feature field.

        Args:
            ref_img: [B, 1, H, W] reference image
            tar_img: [B, 1, H, W] target image

        Returns:
            F_input: [B, N, d] flattened feature field
        """
        # Concatenate along channel dim
        if ref_img.shape[1] == 1 and tar_img.shape[1] == 1:
            x = torch.cat([ref_img, tar_img, tar_img - ref_img], dim=1)  # [B, 3, H, W]
        else:
            x = torch.cat([ref_img, tar_img, tar_img - ref_img], dim=1)

        B, _, H, W = x.shape
        x = self.cnn(x)  # [B, d, H/ds, W/ds]

        # Add position embedding
        _, _, Hf, Wf = x.shape
        pos = self.pos_embed[:, :, :Hf, :Wf]
        x = x + pos

        x = self.proj(x)  # [B, feature_dim, H/ds, W/ds]

        # Flatten spatial dims: [B, N, d]
        x = x.flatten(2).transpose(1, 2)
        return x
