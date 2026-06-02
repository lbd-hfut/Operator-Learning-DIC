"""Decoder for Deformation Inverse Operator (Route B).

Decodes displacement at query points by directly sampling local features
from both SiameseCNN feature maps — no cross-attention bottleneck.

For each query point y:
  f_ref(y) = bilinear_sample(F_ref_2d, y)
  f_tar(y) = bilinear_sample(F_tar_2d, y)
  u(y) = MLP([GFF(y), f_ref, f_tar, f_tar - f_ref])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.gaussian_fourier_features import GaussianFourierFeatureTransform


class InverseDecoder(nn.Module):
    """Local-feature decoder for Route B.

    Samples encoder features from both ref and tar feature maps
    at each query point, concatenates with GFF position encoding,
    and processes through an MLP to predict displacement.
    """

    def __init__(self, feature_dim=256, hidden_dim=512):
        super().__init__()
        self.gff = GaussianFourierFeatureTransform(
            mapping_size=128, scale=2.0, trainable_scale=True,
        )
        # GFF(256) + f_ref(256) + f_tar(256) + f_diff(256) = 1024
        in_dim = self.gff.output_dim + feature_dim * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, query_points, f_ref, f_tar):
        """Decode displacement from SiameseCNN feature maps.

        Args:
            query_points: [B, N_q, 2] normalized coords in [0,1]
            f_ref: [B, d, Hf, Wf] reference image features
            f_tar: [B, d, Hf, Wf] target image features

        Returns:
            u_pred: [B, N_q, 2]
        """
        B, N_q, _ = query_points.shape
        B, d, Hf, Wf = f_ref.shape

        def sample(f_map):
            f_2d = f_map
            grid = query_points * 2.0 - 1.0
            grid = grid.unsqueeze(2)
            out = F.grid_sample(f_2d, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
            return out.squeeze(-1).transpose(1, 2)  # [B, N_q, d]

        f_ref_local = sample(f_ref)
        f_tar_local = sample(f_tar)
        f_diff = f_tar_local - f_ref_local

        pos = self.gff(query_points)
        x = torch.cat([pos, f_ref_local, f_tar_local, f_diff], dim=-1)
        return self.net(x)
