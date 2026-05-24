"""Weight initialization utilities following OFormer convention.

Key insight: initialize attention projection matrices with near-orthogonal
weights and diagonal bias to promote diverse, independent basis functions.
"""
import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = 1.0):
    """Orthogonal weight initialization for Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def xavier_init(module: nn.Module, gain: float = 1.0):
    """Xavier uniform initialization for Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def diagonal_bias_init(module: nn.Module, num_heads: int):
    """Initialize multi-head attention projection with diagonal-like bias.

    Following OFormer: after orthogonal init of weight, set bias so that
    each head gets a distinct bias pattern, promoting diverse basis functions.
    """
    if isinstance(module, nn.Linear) and module.bias is not None:
        dim = module.bias.shape[0]
        head_dim = dim // num_heads
        # Each head gets its own bias scale
        bias = torch.zeros(dim)
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            bias[start:end] = torch.randn(head_dim) * 0.02
        module.bias.data.copy_(bias)


def init_weights(module: nn.Module, num_heads: int = 8):
    """Apply OFormer-style weight initialization to a module hierarchy."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            orthogonal_init(m)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
