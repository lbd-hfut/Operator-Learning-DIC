"""LayerNorm wrapper patterns matching OFormer convention."""
import torch.nn as nn


class PreNorm(nn.Module):
    """Apply LayerNorm before the module, with optional residual connection."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    """Apply module then LayerNorm, with residual connection."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(x + self.fn(x, **kwargs))
