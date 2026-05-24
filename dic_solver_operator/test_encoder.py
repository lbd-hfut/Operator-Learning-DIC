"""Smoke test for DualChannelCNN Encoder."""
import torch
from dic_solver_operator.encoder import DualChannelCNNEncoder


def test_encoder_shapes():
    encoder = DualChannelCNNEncoder(
        in_channels=2,
        channels=(32, 64, 128),
        kernel_size=7,
        n_blocks=2,
        downsample_factor=2,
        feature_dim=128,
    )
    ref = torch.randn(2, 1, 128, 128)
    tar = torch.randn(2, 1, 128, 128)
    out = encoder(ref, tar)

    H, W = 128, 128
    ds = 2 ** encoder.downsample_factor  # 4
    expected_n = (H // ds) * (W // ds)     # 32*32 = 1024
    assert out.shape == (2, expected_n, 128), f"Expected (2, {expected_n}, 128), got {out.shape}"
    print(f"[PASS] test_encoder_shapes: output={out.shape}")


def test_encoder_no_nan():
    encoder = DualChannelCNNEncoder()
    ref = torch.rand(4, 1, 256, 256)
    tar = torch.rand(4, 1, 256, 256)
    out = encoder(ref, tar)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print("[PASS] test_encoder_no_nan")


if __name__ == "__main__":
    test_encoder_shapes()
    test_encoder_no_nan()
    print("All encoder tests passed.")
