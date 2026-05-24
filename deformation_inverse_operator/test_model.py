"""Integration test for InverseOperatorModel."""
import torch
from deformation_inverse_operator.config import InverseOperatorConfig
from deformation_inverse_operator.model import InverseOperatorModel


def test_model_forward():
    config = InverseOperatorConfig(
        siamese_channels=(32, 64, 128),
        siamese_n_blocks=1,
        siamese_downsample=1,
        feature_dim=128,
        num_latent_tokens=64,
        latent_dim=128,
        encoder_self_attn_depth=1,
        fourier_mapping_size=64,
        query_mlp_dim=64,
        attn_heads=4,
        attn_dim_head=32,
        decoder_mlp_dim=64,
        image_size=(128, 128),
    )
    model = InverseOperatorModel(config)

    B, N_q = 2, 50
    ref = torch.rand(B, 1, 128, 128)
    tar = torch.rand(B, 1, 128, 128)
    qpts = torch.rand(B, N_q, 2)

    u_pred = model(ref, tar, qpts)
    assert u_pred.shape == (B, N_q, 2), f"Expected ({B}, {N_q}, 2), got {u_pred.shape}"
    print(f"[PASS] test_model_forward: {u_pred.shape}")


def test_encode_decode_separate():
    config = InverseOperatorConfig(
        siamese_channels=(32, 64, 128),
        siamese_n_blocks=1,
        siamese_downsample=1,
        feature_dim=128,
        num_latent_tokens=64,
        latent_dim=128,
        encoder_self_attn_depth=1,
        fourier_mapping_size=64,
        attn_heads=4,
        attn_dim_head=32,
        query_mlp_dim=64,
        decoder_mlp_dim=64,
        image_size=(64, 64),
    )
    model = InverseOperatorModel(config)

    ref = torch.rand(1, 1, 64, 64)
    tar = torch.rand(1, 1, 64, 64)
    qpts_1 = torch.rand(1, 30, 2)
    qpts_2 = torch.rand(1, 100, 2)

    z = model.encode(ref, tar)
    assert z.shape == (1, config.num_latent_tokens, config.latent_dim)

    u1 = model.decode(qpts_1, z)
    u2 = model.decode(qpts_2, z)
    assert u1.shape == (1, 30, 2)
    assert u2.shape == (1, 100, 2)
    print("[PASS] test_encode_decode_separate")


if __name__ == "__main__":
    test_model_forward()
    test_encode_decode_separate()
    print("All model integration tests passed.")
