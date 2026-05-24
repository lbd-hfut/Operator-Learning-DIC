"""Rotary Position Embedding (RoPE) for 2D coordinates.

Encodes relative spatial position information into attention scores.
Adapted from OFormer's implementation for 2D grid positions.
"""
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """2D rotary position embedding.

    Generates frequency bands for x and y dimensions independently,
    enabling the attention mechanism to be aware of relative 2D positions.
    """

    def __init__(self, dim: int, min_freq: float = 1 / 64):
        super().__init__()
        inv_freq = 1.0 / (
            min_freq ** (torch.arange(0, dim, 4, dtype=torch.float32) / (dim // 2))
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords: torch.Tensor):
        """Compute rotary embeddings for 2D coordinates.

        Args:
            coords: [B, N, 2] normalized coordinates in [0, 1]²

        Returns:
            [B, N, dim] rotary position embeddings
        """
        x, y = coords[..., 0:1], coords[..., 1:2]
        freqs_x = x * self.inv_freq[None, None, :]
        freqs_y = y * self.inv_freq[None, None, :]

        # Interleave x and y frequencies: [cos(fx0), cos(fy0), sin(fx0), sin(fy0), ...]
        emb = torch.cat([freqs_x, freqs_y], dim=-1)
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def apply_2d_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    """Apply 2D rotary position embedding to input tensor.

    Args:
        x: [B, N, dim] input features
        sin: [B, N, dim] precomputed sin values
        cos: [B, N, dim] precomputed cos values

    Returns:
        [B, N, dim] rotated features
    """
    dim_half = x.shape[-1] // 2
    sin = sin[..., :dim_half]
    cos = cos[..., :dim_half]
    x1, x2 = x[..., :dim_half], x[..., dim_half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
