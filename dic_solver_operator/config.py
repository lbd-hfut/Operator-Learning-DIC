"""Configuration for DIC Solver Operator (Route A).

The Solver Operator treats grayscale invariance I_ref(x) = I_tar(x + u(x))
as a shared PDE solved by the operator G: (I_ref, I_tar) -> u.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class SolverOperatorConfig:
    """Configuration for Route A: DIC Solver Operator."""

    # --- Encoder (Dual-Channel CNN) ---
    encoder_in_channels: int = 2          # ref + tar concatenated
    encoder_channels: Tuple[int, ...] = (64, 128, 256)
    encoder_kernel_size: int = 7
    encoder_n_blocks: int = 4             # ResBlocks per stage
    encoder_downsample: int = 2           # total stride factor (2 => H/4, W/4)
    feature_dim: int = 256                # output feature dimension d

    # --- Query Encoder ---
    fourier_mapping_size: int = 128
    fourier_scale: float = 1.0
    fourier_trainable_scale: bool = True
    query_mlp_depth: int = 2
    query_mlp_dim: int = 256

    # --- Cross-Attention Decoder ---
    attn_heads: int = 8
    attn_dim_head: int = 64
    attn_dropout: float = 0.0
    attn_pre_norm: bool = True            # Galerkin-type InstanceNorm
    attn_residual: bool = True
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
    checkpoint_dir: str = "checkpoints/route_a"
    experiment_name: str = "solver_operator"
    use_wandb: bool = False
