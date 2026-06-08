"""Configuration for Route D: ViT-based Transformer DIC Operator.

Loads a frozen ViT-B/16 encoder and learns a query-point decoder
with RoPE, cross-attention, and dual-scale discretized output heads.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


def _make_coarse_bins(n_bins: int = 32, max_val: float = 20.0,
                      min_val: float = 0.01) -> np.ndarray:
    """Build log-spaced coarse bin centers, symmetric around zero.

    Returns [n_bins] float32 array, sorted ascending.
    """
    n_half = n_bins // 2
    # Positive side: log-spaced from min_val to max_val
    pos = np.logspace(np.log10(min_val), np.log10(max_val), n_half)
    # Symmetric + zero center
    bins = np.concatenate([-pos[::-1], [0.0], pos])
    return bins.astype(np.float32)


def _make_fine_bins(n_bins: int = 64, half_range: float = 0.5) -> np.ndarray:
    """Build uniform fine-resolution bin centers.

    Returns [n_bins] float32 array.
    """
    return np.linspace(-half_range, half_range, n_bins, dtype=np.float32)


@dataclass
class VitDICConfig:
    """Configuration for Route D: ViT Transformer DIC Operator."""

    # --- ViT Encoder ---
    vit_model: str = "vit_b_16"            # torchvision ViT variant
    vit_pretrained: bool = True            # load ImageNet weights
    vit_freeze: bool = True                # freeze ViT backbone
    vit_feature_dim: int = 768             # ViT-B/16 hidden dim
    feature_dim: int = 256                 # projected feature dimension
    n_patches: int = 256                   # (256/16)^2 = 256 patch tokens
    image_size: Tuple[int, int] = (256, 256)

    # --- RoPE (Rotary Position Embedding) ---
    rope_dim: int = 256                    # RoPE encoding dimension (must be multiple of 4)
    rope_min_freq: float = 1.0 / 256       # min freq for RoPE

    # --- Decoder Cross-Attention ---
    n_cross_attn_layers: int = 2           # stacked cross-attention blocks per branch (ref / tar)
    attn_heads: int = 8
    attn_dim_head: int = 32                # per-head dim (8×32 = 256 = feature_dim)
    attn_dropout: float = 0.0

    # --- Decoder MLP ---
    decoder_mlp_hidden: int = 512          # hidden dim of fusion MLP
    decoder_mlp_out: int = 256             # output dim before heads

    # --- Output Heads ---
    coarse_n_bins: int = 32                # log-spaced coarse bins
    coarse_max_val: float = 20.0           # max |displacement| for coarse head (px)
    coarse_min_val: float = 0.01           # min bin spacing for coarse head
    fine_n_bins: int = 64                  # uniform fine bins
    fine_half_range: float = 0.5           # fine head covers ±0.5 px

    # --- Loss ---
    fine_loss_weight: float = 10.0         # λ for fine regression loss

    # --- Query Encoding ---
    query_mlp_depth: int = 2               # MLP layers for query token projection
    query_mlp_dim: int = 256

    # --- Training ---
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 2000
    warmup_start_factor: float = 1e-3
    max_epochs: int = 100
    batch_size: int = 4                    # ViT needs more memory
    accumulate_grad_batches: int = 1

    # --- Distributed ---
    use_ddp: bool = False
    local_rank: int = -1

    # --- Logging ---
    log_every: int = 100
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints/route_d"
    experiment_name: str = "vit_dic_operator"
    use_wandb: bool = False

    def __post_init__(self):
        # Pre-compute bin centers
        self.coarse_bin_centers = _make_coarse_bins(
            self.coarse_n_bins, self.coarse_max_val, self.coarse_min_val,
        )
        self.fine_bin_centers = _make_fine_bins(
            self.fine_n_bins, self.fine_half_range,
        )
