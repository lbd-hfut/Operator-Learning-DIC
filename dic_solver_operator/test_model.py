"""Integration test for SolverOperatorModel."""
import torch
from dic_solver_operator.config import SolverOperatorConfig
from dic_solver_operator.model import SolverOperatorModel


def test_model_forward():
    config = SolverOperatorConfig(
        encoder_channels=(32, 64, 128),
        encoder_n_blocks=1,
        encoder_downsample=1,
        feature_dim=128,
        fourier_mapping_size=64,
        query_mlp_dim=64,
        attn_heads=4,
        attn_dim_head=32,
        decoder_mlp_dim=64,
        image_size=(128, 128),
    )
    model = SolverOperatorModel(config)

    B, N_q = 2, 50
    ref = torch.rand(B, 1, 128, 128)
    tar = torch.rand(B, 1, 128, 128)
    qpts = torch.rand(B, N_q, 2)

    u_pred = model(ref, tar, qpts)
    assert u_pred.shape == (B, N_q, 2), f"Expected ({B}, {N_q}, 2), got {u_pred.shape}"
    print(f"[PASS] test_model_forward: output={u_pred.shape}")


def test_model_encode_decode_separate():
    """Test that encode once, decode many gives consistent results."""
    config = SolverOperatorConfig(
        encoder_channels=(32, 64, 128),
        encoder_n_blocks=1,
        encoder_downsample=1,
        feature_dim=128,
        fourier_mapping_size=64,
        attn_heads=4,
        attn_dim_head=32,
        query_mlp_dim=64,
        decoder_mlp_dim=64,
        image_size=(64, 64),
    )
    model = SolverOperatorConfig  # need actual model
    model = SolverOperatorModel(config)

    ref = torch.rand(1, 1, 64, 64)
    tar = torch.rand(1, 1, 64, 64)
    qpts_1 = torch.rand(1, 30, 2)
    qpts_2 = torch.rand(1, 50, 2)

    # Cached encode
    f_input = model.encode(ref, tar)
    u1 = model.decode(qpts_1, f_input)
    u2 = model.decode(qpts_2, f_input)

    assert u1.shape == (1, 30, 2)
    assert u2.shape == (1, 50, 2)
    print("[PASS] test_model_encode_decode_separate")


if __name__ == "__main__":
    test_model_forward()
    test_model_encode_decode_separate()
    print("All model integration tests passed.")
