"""Route D Model: ViT-based Transformer DIC Operator.

VitDICModel wraps a frozen ViT encoder and a learnable Transformer decoder.
Supports encode() / decode() separation for cached inference.
"""

import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder import TransformerDecoder
from .config import VitDICConfig


class VitDICModel(nn.Module):
    """ViT Transformer DIC Operator: G: (I_ref, I_tar) → u.

    Architecture:
      1. ViTEncoder: I_ref → ref_tokens, I_tar → tar_tokens (shared ViT)
      2. TransformerDecoder: query_points + ref_tokens + tar_tokens → u_pred

    Supports:
      - Single forward pass: encode + decode
      - Cached inference: encode once, decode many times
    """

    def __init__(self, config: VitDICConfig):
        super().__init__()
        self.config = config

        # Shared hybrid CNN+ViT encoder for ref and tar
        self.encoder = ViTEncoder(
            feature_dim=config.feature_dim,
            vit_feature_dim=config.vit_feature_dim,
            n_patches=config.n_patches,
            rope_dim=config.rope_dim,
            rope_min_freq=config.rope_min_freq,
            freeze_vit=config.vit_freeze,
            pretrained=config.vit_pretrained,
            cnn_channels=config.cnn_channels if config.use_cnn_frontend else None,
        )

        self.decoder = TransformerDecoder(config)

    def encode(
        self, ref_img: torch.Tensor, tar_img: torch.Tensor,
    ):
        """Encode image pair to ViT token sequences.

        Args:
            ref_img: [B, 1, H, W]
            tar_img: [B, 1, H, W]

        Returns:
            (ref_tokens, tar_tokens): each [B, n_patches, feature_dim]
        """
        ref_tokens = self.encoder(ref_img)
        tar_tokens = self.encoder(tar_img)
        return ref_tokens, tar_tokens

    def decode(
        self, query_points: torch.Tensor,
        ref_tokens: torch.Tensor, tar_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Decode displacement at query points given cached ViT tokens.

        Args:
            query_points: [B, N_q, 2] normalized coords in [0,1]^2
            ref_tokens:   [B, n_patches, feature_dim]
            tar_tokens:   [B, n_patches, feature_dim]

        Returns:
            u_pred: [B, N_q, 2] displacement in pixels
        """
        u_pred, _ = self.decoder(query_points, ref_tokens, tar_tokens)
        return u_pred

    def forward(
        self,
        ref_img: torch.Tensor,
        tar_img: torch.Tensor,
        query_points: torch.Tensor,
    ):
        """Full forward pass.

        Args:
            ref_img:      [B, 1, H, W]
            tar_img:      [B, 1, H, W]
            query_points: [B, N_q, 2] normalized coords in [0,1]^2

        Returns:
            u_pred:       [B, N_q, 2]
            loss_aux:      dict with coarse/fine logits for loss computation
        """
        ref_tokens, tar_tokens = self.encode(ref_img, tar_img)
        return self.decoder(query_points, ref_tokens, tar_tokens)
