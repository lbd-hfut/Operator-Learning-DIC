"""Decoder for Deformation Inverse Operator (Route B).

Decodes displacement at arbitrary query coordinates by cross-attending
to the compact latent code z (rather than the full feature field).

Key advantage over Route A decoder:
  - Complexity O(M·d²) where M << N (M=128 vs N=1024+)
  - z encodes the entire deformation field once; decoding is just lookup
"""
import torch
import torch.nn as nn

from common.gaussian_fourier_features import GaussianFourierFeatureTransform
from common.cross_attention import CrossLinearAttention
from common.feedforward import FeedForward
from common.layer_norm import PostNorm


class InverseDecoder(nn.Module):
    """Operator decoder for Route B.

    Given compact latent code z, decodes displacement at query points
    via learnable cross-attention.

    Args:
        latent_dim: dimension of latent code z
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
        latent_dim: int = 256,
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

        self.gff = GaussianFourierFeatureTransform(
            mapping_size=fourier_mapping_size,
            scale=fourier_scale,
            trainable_scale=fourier_trainable_scale,
        )

        # Query MLP
        query_layers = []
        in_dim = self.gff.output_dim
        for _ in range(query_mlp_depth - 1):
            query_layers.extend([nn.Linear(in_dim, query_mlp_dim), nn.GELU()])
            in_dim = query_mlp_dim
        query_layers.append(nn.Linear(in_dim, query_mlp_dim))
        self.query_encoder = nn.Sequential(*query_layers)

        # Cross-attention: query attends to latent code z
        self.cross_attn = CrossLinearAttention(
            query_dim=query_mlp_dim,
            context_dim=latent_dim,
            dim=decoder_mlp_dim,
            heads=attn_heads,
            dim_head=attn_dim_head,
            dropout=attn_dropout,
            pre_norm=attn_pre_norm,
            residual=attn_residual,
        )

        # Output MLP
        output_layers = []
        in_dim = decoder_mlp_dim
        for _ in range(decoder_mlp_depth - 1):
            output_layers.append(PostNorm(in_dim, FeedForward(in_dim)))
        output_layers.append(nn.Linear(in_dim, 2))
        self.output_head = nn.Sequential(*output_layers)

    def forward(
        self,
        query_points: torch.Tensor,
        latent_code: torch.Tensor,
    ) -> torch.Tensor:
        """Decode displacement at query points from latent code.

        Args:
            query_points: [B, N_q, 2] normalized coordinates in [0,1]²
            latent_code: [B, M, d] compact deformation encoding

        Returns:
            u_pred: [B, N_q, 2] predicted displacement in pixels
        """
        q = self.gff(query_points)
        q = self.query_encoder(q)
        q = self.cross_attn(q, latent_code)
        u_pred = self.output_head(q)
        return u_pred
