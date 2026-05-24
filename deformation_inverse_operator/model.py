"""Full Deformation Inverse Operator Model (Route B).

Combines SiameseCNN + DifferentialCrossAttention encoder
with cross-attention decoder over compact latent code z.
"""
import torch
import torch.nn as nn

from .encoder import SiameseCNNEncoder, DifferentialCrossAttentionEncoder
from .decoder import InverseDecoder
from .config import InverseOperatorConfig


class InverseOperatorModel(nn.Module):
    """Deformation Inverse Operator: G: (I_ref, I_tar) -> u.

    Architecture:
      1. SiameseCNN: I_ref -> F_ref, I_tar -> F_tar
      2. DifferentialCrossAttention: (F_ref, F_tar) -> z [B, M, d]
      3. InverseDecoder: (query_points, z) -> u_pred

    The compact latent code z encodes the full-field deformation state,
    enabling efficient querying at arbitrary coordinates without
    accessing the original feature maps.
    """

    def __init__(self, config: InverseOperatorConfig):
        super().__init__()
        self.config = config

        # Siamese CNN: shared weights for ref and tar
        if config.share_weights:
            self.cnn = SiameseCNNEncoder(
                in_channels=1,
                channels=config.siamese_channels,
                n_blocks=config.siamese_n_blocks,
                downsample_factor=config.siamese_downsample,
                feature_dim=config.feature_dim,
            )
            self.cnn_ref = self.cnn
            self.cnn_tar = self.cnn
        else:
            self.cnn_ref = SiameseCNNEncoder(
                in_channels=1,
                channels=config.siamese_channels,
                n_blocks=config.siamese_n_blocks,
                downsample_factor=config.siamese_downsample,
                feature_dim=config.feature_dim,
            )
            self.cnn_tar = SiameseCNNEncoder(
                in_channels=1,
                channels=config.siamese_channels,
                n_blocks=config.siamese_n_blocks,
                downsample_factor=config.siamese_downsample,
                feature_dim=config.feature_dim,
            )

        # Latent encoding via differential cross-attention
        self.latent_encoder = DifferentialCrossAttentionEncoder(
            feature_dim=config.feature_dim,
            num_latent_tokens=config.num_latent_tokens,
            latent_dim=config.latent_dim,
            cross_attn_depth=config.encoder_cross_attn_depth,
            self_attn_depth=config.encoder_self_attn_depth,
            attn_heads=config.attn_heads,
            attn_dim_head=config.attn_dim_head,
            attn_dropout=config.attn_dropout,
            attn_pre_norm=config.attn_pre_norm,
            attn_residual=config.attn_residual,
        )

        # Decoder: queries latent code at arbitrary coordinates
        self.decoder = InverseDecoder(
            latent_dim=config.latent_dim,
            fourier_mapping_size=config.fourier_mapping_size,
            fourier_scale=config.fourier_scale,
            fourier_trainable_scale=config.fourier_trainable_scale,
            query_mlp_depth=config.query_mlp_depth,
            query_mlp_dim=config.query_mlp_dim,
            attn_heads=config.attn_heads,
            attn_dim_head=config.attn_dim_head,
            attn_dropout=config.attn_dropout,
            attn_pre_norm=config.attn_pre_norm,
            attn_residual=config.attn_residual,
            decoder_mlp_depth=config.decoder_mlp_depth,
            decoder_mlp_dim=config.decoder_mlp_dim,
        )

    def encode(
        self, ref_img: torch.Tensor, tar_img: torch.Tensor
    ) -> torch.Tensor:
        """Encode image pair to compact latent code.

        Call once per image pair; then decode at arbitrary query points.

        Args:
            ref_img: [B, 1, H, W]
            tar_img: [B, 1, H, W]

        Returns:
            z: [B, M, d] deformation latent code
        """
        f_ref = self.cnn_ref(ref_img)     # [B, N, feature_dim]
        f_tar = self.cnn_tar(tar_img)     # [B, N, feature_dim]
        z = self.latent_encoder(f_ref, f_tar)
        return z

    def decode(
        self, query_points: torch.Tensor, latent_code: torch.Tensor
    ) -> torch.Tensor:
        """Decode displacement at query points from latent code.

        Args:
            query_points: [B, N_q, 2] normalized coordinates in [0,1]²
            latent_code: [B, M, d] from encode()

        Returns:
            u_pred: [B, N_q, 2] predicted displacement in pixels
        """
        return self.decoder(query_points, latent_code)

    def forward(
        self,
        ref_img: torch.Tensor,
        tar_img: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            ref_img: [B, 1, H, W]
            tar_img: [B, 1, H, W]
            query_points: [B, N_q, 2]

        Returns:
            u_pred: [B, N_q, 2]
        """
        z = self.encode(ref_img, tar_img)
        return self.decode(query_points, z)
