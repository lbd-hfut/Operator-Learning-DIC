"""Configuration for U-Net DIC Method (Route C).

Traditional deep-learning DIC: a U-Net takes concatenated [I_ref, I_tar]
and directly predicts a dense displacement field u [2, H, W].

Unlike Route A/B which are query-point operators, Route C is a dense
image-to-image translation method.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class UnetDICConfig:
    """Configuration for Route C: U-Net DIC."""

    # --- U-Net Architecture ---
    in_channels: int = 2                # ref + tar concatenated
    out_channels: int = 2               # u_x, u_y
    base_channels: int = 64             # first-stage channel count
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8, 16)  # per-stage multiplier
    n_blocks_per_stage: int = 2         # DoubleConv blocks per encoder stage
    use_group_norm: bool = True         # GroupNorm (like Route A/B) instead of BatchNorm

    # --- Training ---
    image_size: Tuple[int, int] = (256, 256)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 0
    warmup_start_factor: float = 1e-3
    max_steps: int = 100000
    batch_size: int = 8

    # --- Loss ---
    data_loss_type: str = "mse"

    # --- Distributed ---
    use_ddp: bool = False
    local_rank: int = -1

    # --- Logging ---
    log_every: int = 100
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints/route_c"
    experiment_name: str = "unet_dic"
    use_wandb: bool = False
