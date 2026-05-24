"""Smoke test for Route B Decoder."""
import torch
from deformation_inverse_operator.decoder import InverseDecoder


def test_decoder_shape():
    decoder = InverseDecoder(
        latent_dim=128,
        fourier_mapping_size=64,
        query_mlp_dim=128,
        attn_heads=4,
        attn_dim_head=32,
        decoder_mlp_dim=128,
    )
    qpts = torch.rand(2, 100, 2)
    z = torch.randn(2, 64, 128)  # [B, M, d]
    u_pred = decoder(qpts, z)
    assert u_pred.shape == (2, 100, 2)
    print(f"[PASS] test_decoder_shape: {u_pred.shape}")


def test_decoder_variable_queries():
    """Decoder should handle different query counts with the same latent code."""
    decoder = InverseDecoder(
        latent_dim=128,
        fourier_mapping_size=64,
        attn_heads=4,
        attn_dim_head=32,
    )
    z = torch.randn(1, 64, 128)  # one latent code

    for n_q in [10, 50, 200]:
        qpts = torch.rand(1, n_q, 2)
        u_pred = decoder(qpts, z)
        assert u_pred.shape == (1, n_q, 2)
        print(f"  query_count={n_q:3d} -> u_pred shape={u_pred.shape}")
    print("[PASS] test_decoder_variable_queries")


if __name__ == "__main__":
    test_decoder_shape()
    test_decoder_variable_queries()
    print("All decoder tests passed.")
