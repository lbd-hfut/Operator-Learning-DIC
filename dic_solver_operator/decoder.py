"""Decoder for DIC Solver Operator (Route A).

Decodes displacement values at arbitrary query coordinates by
cross-attending to the feature field F_input from the encoder.

For each query point y:
  q(y) = MLP(GFF(y))
  u(y) = CrossAttn(q(y), F_input) -> MLP -> (u_x, u_y)
"""
import torch
import torch.nn as nn

from common.gaussian_fourier_features import GaussianFourierFeatureTransform
from common.cross_attention import CrossLinearAttention
from common.feedforward import FeedForward
from common.layer_norm import PostNorm


class SolverDecoder(nn.Module):
    """Operator decoder for Route A.

    Given the feature field F_input, decodes displacement at query points
    via learnable cross-attention.

    Args:
        feature_dim: dimension of F_input features
        fourier_mapping_size: GFF random features
        fourier_scale: sigma for GFF
        query_mlp_depth: MLP depth for query encoding
        query_mlp_dim: hidden dim for query MLP
        attn_heads: number of attention heads
        attn_dim_head: dimension per head
        attn_dropout: attention dropout
        attn_pre_norm: Galerkin-type InstanceNorm on K,V
        decoder_mlp_depth: MLP depth after attention
        decoder_mlp_dim: hidden dim for decoder MLP
    """

    def __init__(
        self,
        feature_dim: int = 256,
        fourier_mapping_size: int = 128,
        fourier_scale: float = 10.0,
        fourier_trainable_scale: bool = True,
        query_mlp_depth: int = 2,
        query_mlp_dim: int = 256,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_dropout: float = 0.0,
        attn_pre_norm: bool = True,
        attn_residual: bool = True,
        decoder_mlp_depth: int = 2,
        decoder_mlp_dim: int = 256,
    ):
        super().__init__()

        # Coordinate encoding
        self.gff = GaussianFourierFeatureTransform(
            mapping_size=fourier_mapping_size,
            scale=fourier_scale,
            trainable_scale=fourier_trainable_scale,
        )

        # Query MLP: GFF output -> query embedding
        query_layers = []
        in_dim = self.gff.output_dim
        for _ in range(query_mlp_depth - 1):
            query_layers.extend([
                nn.Linear(in_dim, query_mlp_dim),
                nn.GELU(),
            ])
            in_dim = query_mlp_dim
        query_layers.append(nn.Linear(in_dim, query_mlp_dim))
        self.query_encoder = nn.Sequential(*query_layers)

        # Cross-attention: query attends to F_input
        self.cross_attn = CrossLinearAttention(
            query_dim=query_mlp_dim,
            context_dim=feature_dim,
            dim=decoder_mlp_dim,
            heads=attn_heads,
            dim_head=attn_dim_head,
            dropout=attn_dropout,
            pre_norm=attn_pre_norm,
            residual=attn_residual,
        )

        # Output MLP: attention output -> displacement
        output_layers = []
        in_dim = decoder_mlp_dim
        for _ in range(decoder_mlp_depth - 1):
            output_layers.append(FeedForward(in_dim))
        output_layers.append(nn.Linear(in_dim, 2))
        self.output_head = nn.Sequential(*output_layers)

    def forward(
        self,
        query_points: torch.Tensor,
        f_input: torch.Tensor,
    ) -> torch.Tensor:
        """Decode displacement at query points.

        Args:
            query_points: [B, N_q, 2] normalized coordinates in [0,1]²
            f_input: [B, N_kv, d] feature field from encoder

        Returns:
            u_pred: [B, N_q, 2] predicted displacement in pixels
        """
        B, N_q, _ = query_points.shape

        # Encode query coordinates
        q = self.gff(query_points)          # [B, N_q, 2*mapping_size]
        q = self.query_encoder(q)           # [B, N_q, query_mlp_dim]

        # Cross-attend to feature field
        q = self.cross_attn(q, f_input)     # [B, N_q, decoder_mlp_dim]

        # Predict displacement
        u_pred = self.output_head(q)        # [B, N_q, 2]
        return u_pred
