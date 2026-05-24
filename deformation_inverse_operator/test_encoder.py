"""Smoke test for Route B Encoder."""
import torch
from deformation_inverse_operator.encoder import (
    SiameseCNNEncoder,
    DifferentialCrossAttentionEncoder,
)


def test_siamese_cnn_shape():
    cnn = SiameseCNNEncoder(
        in_channels=1,
        channels=(32, 64, 128),
        n_blocks=2,
        downsample_factor=2,
        feature_dim=128,
    )
    img = torch.randn(2, 1, 128, 128)
    out = cnn(img)

    H, W = 128, 128
    ds = 2 ** cnn.downsample_factor
    expected_n = (H // ds) * (W // ds)
    assert out.shape == (2, expected_n, 128), f"Expected (2, {expected_n}, 128), got {out.shape}"
    print(f"[PASS] test_siamese_cnn_shape: {out.shape}")


def test_differential_cross_attention():
    enc = DifferentialCrossAttentionEncoder(
        feature_dim=128,
        num_latent_tokens=64,
        latent_dim=128,
        attn_heads=4,
        attn_dim_head=32,
    )
    f_ref = torch.randn(2, 256, 128)  # [B, N, d]
    f_tar = torch.randn(2, 256, 128)
    z = enc(f_ref, f_tar)
    assert z.shape == (2, 64, 128), f"Expected (2, 64, 128), got {z.shape}"
    print(f"[PASS] test_differential_cross_attention: z={z.shape}")


if __name__ == "__main__":
    test_siamese_cnn_shape()
    test_differential_cross_attention()
    print("All encoder tests passed.")
