"""Full Deformation Inverse Operator Model (Route B).

SiameseCNN encoder -> local feature sampling decoder (no cross-attention).
The DifferentialCrossAttention latent encoder was removed — it caused
mode collapse due to the same K^T@V bottleneck as Route A.
"""
import torch
import torch.nn as nn

from .encoder import SiameseCNNEncoder
from .decoder import InverseDecoder
from .config import InverseOperatorConfig


class InverseOperatorModel(nn.Module):
    """Deformation Inverse Operator: G: (I_ref, I_tar) -> u.

    Architecture:
      1. SiameseCNN: I_ref -> F_ref, I_tar -> F_tar
      2. InverseDecoder: samples F_ref, F_tar at query coords -> MLP -> u
    """

    def __init__(self, config: InverseOperatorConfig):
        super().__init__()
        self.config = config

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

        self.decoder = InverseDecoder(feature_dim=config.feature_dim)

    def encode(self, ref_img, tar_img):
        """Encode image pair to feature maps."""
        f_ref = self.cnn_ref(ref_img)
        f_tar = self.cnn_tar(tar_img)
        return f_ref, f_tar

    def decode(self, query_points, f_ref, f_tar):
        """Decode displacement at query points from feature maps."""
        return self.decoder(query_points, f_ref, f_tar)

    def forward(self, ref_img, tar_img, query_points):
        f_ref, f_tar = self.encode(ref_img, tar_img)
        return self.decode(query_points, f_ref, f_tar)
