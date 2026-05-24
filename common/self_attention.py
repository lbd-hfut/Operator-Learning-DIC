"""OFormer-style Galerkin linear self-attention.

Same mechanism as CrossLinearAttention but Q, K, V all come from the same input.
Used by Route B encoder for processing latent tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """Galerkin-type linear self-attention.

    Computes Z = Q @ (K^T @ V) / n where Q, K, V all project from the same input.

    Args:
        dim: input/output dimension
        heads: number of attention heads
        dim_head: dimension per head
        dropout: attention dropout rate
        pre_norm: if True, applies InstanceNorm to K, V columns (Galerkin type)
        residual: if True, adds residual connection
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        pre_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.pre_norm = pre_norm
        self.residual = residual

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        if pre_norm:
            self.k_norm = nn.InstanceNorm1d(inner_dim)
            self.v_norm = nn.InstanceNorm1d(inner_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Self-attention on input x.

        Args:
            x: [B, N, dim] input features
            mask: optional [B, N] mask

        Returns:
            [B, N, dim] attended features
        """
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(q.shape[0], q.shape[1], self.heads, self.dim_head).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.heads, self.dim_head).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, self.dim_head).transpose(1, 2)

        if self.pre_norm:
            B, H, N, D = k.shape
            k = k.reshape(B, H * D, N)
            k = self.k_norm(k)
            k = k.reshape(B, H, N, D)

            v = v.reshape(B, H * D, N)
            v = self.v_norm(v)
            v = v.reshape(B, H, N, D)

        # Galerkin: compute K^T @ V first
        k_t = k.transpose(-1, -2)
        kv = torch.matmul(k_t, v)
        out = torch.matmul(q, kv) / k.shape[2]

        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)

        if self.residual:
            out = out + x

        return out
