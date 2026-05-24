"""Smoke test for Solver Decoder."""
import torch
from dic_solver_operator.decoder import SolverDecoder


def test_decoder_shapes():
    decoder = SolverDecoder(
        feature_dim=128,
        fourier_mapping_size=64,
        query_mlp_dim=128,
        attn_heads=4,
        attn_dim_head=32,
        decoder_mlp_dim=128,
    )
    B, N_q, N_kv = 2, 100, 256
    qpts = torch.rand(B, N_q, 2)
    f_input = torch.randn(B, N_kv, 128)

    u_pred = decoder(qpts, f_input)
    assert u_pred.shape == (B, N_q, 2), f"Expected ({B}, {N_q}, 2), got {u_pred.shape}"
    print(f"[PASS] test_decoder_shapes: output={u_pred.shape}")


def test_decoder_no_nan():
    decoder = SolverDecoder()
    qpts = torch.rand(4, 200, 2)
    f_input = torch.randn(4, 1024, 256)
    u_pred = decoder(qpts, f_input)
    assert not torch.isnan(u_pred).any()
    assert not torch.isinf(u_pred).any()
    print("[PASS] test_decoder_no_nan")


if __name__ == "__main__":
    test_decoder_shapes()
    test_decoder_no_nan()
    print("All decoder tests passed.")
