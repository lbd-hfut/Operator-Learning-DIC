"""Encoder for Deformation Inverse Operator (Route B).

Two-stage encoding:
  1. SiameseCNN: separately encodes I_ref -> F_ref, I_tar -> F_tar
  2. DifferentialCrossAttention: learnable latent queries z_0 sequentially
     cross-attend to F_ref (to understand reference texture) then F_tar
     (to find correspondences and inject displacement information).
"""
import torch
import torch.nn as nn

from common.cross_attention import CrossLinearAttention
from common.self_attention import LinearAttention
from common.feedforward import FeedForward
from common.layer_norm import PostNorm


class ResBlock(nn.Module):
    """Simple residual block for Siamese CNN."""

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


class SiameseCNNEncoder(nn.Module):
    """Weight-shared CNN that encodes a single image to features.

    Used separately for I_ref and I_tar to produce F_ref and F_tar.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple = (64, 128, 256),
        n_blocks: int = 4,
        downsample_factor: int = 2,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        layers = []
        current_ch = in_channels
        layers.append(
            nn.Conv2d(current_ch, channels[0], 7, stride=1, padding=3, bias=False)
        )
        layers.append(nn.GroupNorm(8, channels[0]))
        layers.append(nn.GELU())
        current_ch = channels[0]

        for stage_idx, ch in enumerate(channels):
            stride = 2 if stage_idx < downsample_factor else 1
            for _ in range(n_blocks):
                layers.append(ResBlock(current_ch, ch, stride=stride))
                current_ch = ch
                stride = 1

        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Conv2d(current_ch, feature_dim, 1)

        # Position embedding
        max_hw = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_dim, max_hw, max_hw))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        x = self.cnn(img)
        _, _, Hf, Wf = x.shape
        pos = self.pos_embed[:, :, :Hf, :Wf]
        x = x + pos
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, feature_dim]
        return x


class DifferentialCrossAttentionEncoder(nn.Module):
    """Encode F_ref and F_tar into compact latent code z via differential cross-attention.

    The differential encoding process:
      z_0: learnable latent queries [M, d]
      z_1 = CrossAttn(z_0, F_ref)   ... understand reference texture layout
      z_2 = CrossAttn(z_1, F_tar)   ... find correspondences in target

    The difference z_2 - z_0 encodes the deformation (what moved between images).

    Args:
        feature_dim: dimension of F_ref / F_tar
        num_latent_tokens: M, number of latent queries
        latent_dim: d, latent token dimension
        cross_attn_depth: number of sequential cross-attention layers
        self_attn_depth: number of self-attention refinement layers on z
        attn_heads, attn_dim_head, attn_dropout, attn_pre_norm: attention params
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_latent_tokens: int = 128,
        latent_dim: int = 256,
        cross_attn_depth: int = 1,
        self_attn_depth: int = 2,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_dropout: float = 0.0,
        attn_pre_norm: bool = True,
        attn_residual: bool = True,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.latent_dim = latent_dim

        # Learnable latent queries
        self.latent_queries = nn.Parameter(torch.randn(1, num_latent_tokens, latent_dim) * 0.02)

        # Cross-attention: latent -> F_ref, then latent -> F_tar
        self.cross_attn_ref = CrossLinearAttention(
            query_dim=latent_dim,
            context_dim=feature_dim,
            dim=latent_dim,
            heads=attn_heads,
            dim_head=attn_dim_head,
            dropout=attn_dropout,
            pre_norm=attn_pre_norm,
            residual=attn_residual,
        )

        self.cross_attn_tar = CrossLinearAttention(
            query_dim=latent_dim,
            context_dim=feature_dim,
            dim=latent_dim,
            heads=attn_heads,
            dim_head=attn_dim_head,
            dropout=attn_dropout,
            pre_norm=attn_pre_norm,
            residual=attn_residual,
        )

        # Self-attention refinement on latent tokens
        self.self_attn_layers = nn.ModuleList([
            nn.ModuleList([
                PostNorm(latent_dim, LinearAttention(
                    dim=latent_dim,
                    heads=attn_heads,
                    dim_head=attn_dim_head,
                    dropout=attn_dropout,
                    pre_norm=attn_pre_norm,
                    residual=True,
                )),
                PostNorm(latent_dim, FeedForward(latent_dim)),
            ])
            for _ in range(self_attn_depth)
        ])

    def forward(
        self,
        f_ref: torch.Tensor,
        f_tar: torch.Tensor,
    ) -> torch.Tensor:
        """Encode feature maps to latent code.

        Args:
            f_ref: [B, N, feature_dim] reference image features
            f_tar: [B, N, feature_dim] target image features

        Returns:
            z: [B, M, latent_dim] compact deformation latent code
        """
        B = f_ref.shape[0]
        z = self.latent_queries.expand(B, -1, -1)

        # Differential encoding: ref first, then tar
        z = self.cross_attn_ref(z, f_ref)   # Understand reference texture
        z = self.cross_attn_tar(z, f_tar)    # Find correspondences in target

        # Self-attention refinement
        for self_attn, ffn in self.self_attn_layers:
            z = self_attn(z)
            z = ffn(z)

        return z
