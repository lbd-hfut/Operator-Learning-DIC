"""Configuration for Deformation Inverse Operator (Route B).

The Inverse Operator learns the inverse mapping of the implicit grayscale
invariance constraint: G: (I_ref, I_tar) -> u.

Compared to Route A, key differences:
  - Siamese CNN (separate encoding of I_ref and I_tar)
  - Learnable latent queries that cross-attend sequentially (F_ref -> F_tar)
  - Compact latent code z [B, M, d] instead of full feature field
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class InverseOperatorConfig:
    """Configuration for Route B: Deformation Inverse Operator."""

    # --- Siamese CNN Encoder ---
    siamese_channels: Tuple[int, ...] = (64, 128, 256)
    siamese_n_blocks: int = 4
    siamese_downsample: int = 2        # total stride factor
    share_weights: bool = True         # Siamese: share CNN weights for ref and tar
    feature_dim: int = 256

    # --- Latent Query Encoder ---
    num_latent_tokens: int = 128       # M: number of latent queries
    latent_dim: int = 256              # d: latent token dimension
    encoder_cross_attn_depth: int = 1  # number of cross-attention layers
    encoder_self_attn_depth: int = 2   # self-attention refinement on z

    # --- Query Decoder ---
    fourier_mapping_size: int = 128
    fourier_scale: float = 1.0
    fourier_trainable_scale: bool = True
    query_mlp_depth: int = 2
    query_mlp_dim: int = 256

    # --- Cross-Attention (shared for both encoder and decoder) ---
    attn_heads: int = 8
    attn_dim_head: int = 64
    attn_dropout: float = 0.0
    attn_pre_norm: bool = True
    attn_residual: bool = True

    # --- Decoder MLP ---
    decoder_mlp_depth: int = 2
    decoder_mlp_dim: int = 256

    # --- Training ---
    image_size: Tuple[int, int] = (256, 256)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 8
    accumulate_grad_batches: int = 1

    # --- Loss ---
    data_loss_type: str = "relative_l2"
    lambda_reg: float = 0.0
    reg_loss_type: str = "none"

    # --- Distributed ---
    use_ddp: bool = False
    local_rank: int = -1

    # --- Logging ---
    log_every: int = 100
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints/route_b"
    experiment_name: str = "inverse_operator"
    use_wandb: bool = False
