"""Decoder for DIC Solver Operator (Route A).

Decodes displacement at query points using only local encoder features
(no cross-attention) + positional encoding — a per-point MLP decoder.

For each query point y:
  f_local(y) = bilinear_sample(F_2d, y)
  u(y) = MLP([GFF(y), f_local(y)])
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.gaussian_fourier_features import GaussianFourierFeatureTransform


class SimpleLocalDecoder(nn.Module):
    """Per-point decoder: samples local encoder features + position → MLP → u.

    No cross-attention — tests whether the encoder alone can produce
    displacement-relevant local features at each spatial position.
    """

    def __init__(self, feature_dim=256, pos_dim=256, hidden_dim=512):
        super().__init__()
        self.gff = GaussianFourierFeatureTransform(
            mapping_size=128, scale=2.0, trainable_scale=True,
        )
        self.net = nn.Sequential(
            nn.Linear(self.gff.output_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, query_points, f_input):
        B, N_q, _ = query_points.shape
        _, N_kv, d = f_input.shape
        Hf = Wf = int(math.sqrt(N_kv))

        # Sample local encoder feature at each query point
        f_2d = f_input.transpose(1, 2).reshape(B, d, Hf, Wf)
        grid = query_points * 2.0 - 1.0
        grid = grid.unsqueeze(2)
        f_local = F.grid_sample(f_2d, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        f_local = f_local.squeeze(-1).transpose(1, 2)  # [B, N_q, d]

        # Position encoding
        pos = self.gff(query_points)  # [B, N_q, 2*mapping_size]

        # MLP: position + local feature → displacement
        x = torch.cat([pos, f_local], dim=-1)
        return self.net(x)


class SolverDecoder(SimpleLocalDecoder):
    """Alias for backward compatibility — wraps SimpleLocalDecoder."""
    pass
