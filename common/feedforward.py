"""Feed-forward network blocks with gated activations."""
import torch.nn as nn


class GeGELU(nn.Module):
    """Gated GELU activation: output = x * GELU(x), halving the feature dim."""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)


class FeedForward(nn.Module):
    """Two-layer MLP with gated GELU and residual connection.

    Matches OFormer convention: Linear -> GeGELU -> Linear -> residual.
    """

    def __init__(self, dim: int, expansion_factor: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor * 2)  # *2 for gate split
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
