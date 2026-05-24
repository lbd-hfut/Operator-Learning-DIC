"""Gaussian Fourier Feature coordinate encoding.

Maps 2D coordinates to high-frequency features via random Fourier projections,
enabling the decoder to represent fine spatial variations in displacement fields.
"""
import torch
import torch.nn as nn
import math


class GaussianFourierFeatureTransform(nn.Module):
    """Gaussian random Fourier feature mapping for 2D coordinates.

    gamma(x) = [cos(2π x B), sin(2π x B)]
    where B ~ N(0, σ²) is a random Gaussian matrix.

    Args:
        mapping_size: number of random Fourier features (half the output dim)
        scale: std of the Gaussian initialization of B; controls frequency bandwidth
        trainable_scale: if True, learns a scalar multiplier on B
    """

    def __init__(
        self,
        mapping_size: int = 128,
        scale: float = 1.0,
        trainable_scale: bool = True,
    ):
        super().__init__()
        self.mapping_size = mapping_size
        self._scale = scale

        B = torch.randn(2, mapping_size) * scale
        self.register_buffer("B", B)

        if trainable_scale:
            self.scale_factor = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale_factor", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map coordinates to Fourier features.

        Args:
            x: [B, N, 2] normalized coordinates in [0, 1]²

        Returns:
            [B, N, 2 * mapping_size] Fourier features
        """
        B = self.B * self.scale_factor
        proj = 2 * math.pi * torch.matmul(x, B)  # [B, N, mapping_size]
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self.mapping_size
