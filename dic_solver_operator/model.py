"""Full DIC Solver Operator Model (Route A).

DualChannelCNN encoder -> local feature sampling decoder (no cross-attention).
Supports separate encode()/decode() for flexible inference.
"""
import torch
import torch.nn as nn

from .encoder import DualChannelCNNEncoder
from .decoder import SolverDecoder
from .config import SolverOperatorConfig


class SolverOperatorModel(nn.Module):
    """DIC Solver Operator: G: (I_ref, I_tar) -> u.

    Architecture:
      1. DualChannelCNN encodes [I_ref, I_tar] -> F_input [B, N, d]
      2. SolverDecoder queries F_input at coordinates -> u_pred [B, N_q, 2]

    Supports:
      - Single forward pass: encode + decode
      - Cached inference: encode once, decode many times with different query sets
    """

    def __init__(self, config: SolverOperatorConfig):
        super().__init__()
        self.config = config

        self.encoder = DualChannelCNNEncoder(
            in_channels=config.encoder_in_channels,
            channels=config.encoder_channels,
            kernel_size=config.encoder_kernel_size,
            n_blocks=config.encoder_n_blocks,
            downsample_factor=config.encoder_downsample,
            feature_dim=config.feature_dim,
        )

        self.decoder = SolverDecoder(
            feature_dim=config.feature_dim,
        )

    def encode(
        self, ref_img: torch.Tensor, tar_img: torch.Tensor
    ) -> torch.Tensor:
        """Encode image pair to feature field.

        Call once per image pair; then decode at arbitrary query points.
        """
        return self.encoder(ref_img, tar_img)

    def decode(
        self, query_points: torch.Tensor, f_input: torch.Tensor
    ) -> torch.Tensor:
        """Decode displacement at query points given cached feature field."""
        return self.decoder(query_points, f_input)

    def forward(
        self,
        ref_img: torch.Tensor,
        tar_img: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: encode image pair and decode at query points.

        Args:
            ref_img: [B, 1, H, W]
            tar_img: [B, 1, H, W]
            query_points: [B, N_q, 2] normalized coordinates in [0,1]²

        Returns:
            u_pred: [B, N_q, 2] predicted displacement in pixels
        """
        f_input = self.encode(ref_img, tar_img)
        return self.decode(query_points, f_input)
