"""OFormer-style Galerkin linear cross-attention.

Mathematical interpretation:
  Z = Q @ (K^T @ V) / n

Each column of K, V is a learnable basis function evaluated at discrete points.
InstanceNorm on K/V columns normalizes ||basis_j||_2 = 1.

Complexity: O(N·d²) instead of O(N²·d) for standard softmax attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotary_embedding import apply_2d_rotary_pos_emb


class CrossLinearAttention(nn.Module):
    """Galerkin-type linear cross-attention.

    Query attends to key-value pairs via matrix-associative linear attention.
    Supports RoPE for relative position encoding.

    Args:
        query_dim: dimension of query input
        context_dim: dimension of key/value input (context)
        dim: internal attention dimension
        heads: number of attention heads
        dim_head: dimension per head
        dropout: attention dropout rate
        use_rope: if True, applies rotary position embedding to query and key
        pre_norm: if True, applies InstanceNorm to K, V columns (Galerkin type)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        dim: int = 256,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = False,
        pre_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope
        self.pre_norm = pre_norm
        self.residual = residual

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        if residual:
            self.proj_res = (
                nn.Linear(query_dim, dim) if query_dim != dim else nn.Identity()
            )

        if pre_norm:
            self.k_norm = nn.InstanceNorm1d(inner_dim)
            self.v_norm = nn.InstanceNorm1d(inner_dim)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        pos_emb: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Cross-attention from query x to context z.

        Args:
            x: [B, N_q, query_dim] query features
            z: [B, N_kv, context_dim] key/value context features
            pos_emb: optional [B, N, dim_head] or tuple of (sin, cos) for RoPE
            mask: optional [B, N_q, N_kv] attention mask

        Returns:
            [B, N_q, dim] attended features
        """
        q = self.to_q(x)  # [B, N_q, inner_dim]
        k = self.to_k(z)  # [B, N_kv, inner_dim]
        v = self.to_v(z)  # [B, N_kv, inner_dim]

        # Reshape for multi-head: [B, heads, N, dim_head]
        q = q.view(q.shape[0], q.shape[1], self.heads, self.dim_head).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.heads, self.dim_head).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, self.dim_head).transpose(1, 2)

        # Galerkin-type InstanceNorm on K, V columns (per basis function)
        if self.pre_norm:
            # InstanceNorm1d expects [B, C, L]; we have [B, heads, N, dim_head]
            B, H, N_kv, D = k.shape
            k = k.reshape(B, H * D, N_kv)
            k = self.k_norm(k)
            k = k.reshape(B, H, N_kv, D)

            v = v.reshape(B, H * D, N_kv)
            v = self.v_norm(v)
            v = v.reshape(B, H, N_kv, D)

        # Linear attention: Q @ (K^T @ V) / N_kv
        # Following Galerkin type: compute K^T @ V first
        k_t = k.transpose(-1, -2)  # [B, heads, dim_head, N_kv]
        kv = torch.matmul(k_t, v)  # [B, heads, dim_head, dim_head]
        out = torch.matmul(q, kv) / k.shape[2]  # [B, heads, N_q, dim_head]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)

        if self.residual:
            out = out + self.proj_res(x)

        return out
