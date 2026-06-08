"""Transformer Decoder for Route D.

RoPE-encoded query tokens attend to ref and tar ViT feature tokens
via softmax cross-attention, then fused with feature-diff through an
MLP into dual-scale displacement output heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.rotary_embedding import RotaryEmbedding, apply_2d_rotary_pos_emb


# ---------------------------------------------------------------------------
# Displacement Head — LLM-style learnable bin embeddings
# ---------------------------------------------------------------------------

class DisplacementHead(nn.Module):
    """LLM-style output head: hidden @ bin_emb.T → logits → softmax → expectation.

    Args:
        d_model: input feature dimension
        bin_centers: [n_bins] float32 array of bin center values (fixed)
        with_classification: if True, also compute CE-compatible logits
    """

    def __init__(self, d_model: int, bin_centers: np.ndarray):
        super().__init__()
        n_bins = len(bin_centers)
        self.register_buffer("bin_centers", torch.from_numpy(bin_centers))
        self.bin_embeddings = nn.Parameter(torch.randn(n_bins, d_model) * 0.02)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, N_q, d_model]

        Returns:
            u: [B, N_q] expected displacement
            logits: [B, N_q, n_bins] classification logits (for CE loss)
        """
        logits = torch.matmul(x, self.bin_embeddings.T)  # [B, N_q, n_bins]
        probs = F.softmax(logits, dim=-1)
        u = torch.matmul(probs, self.bin_centers)         # [B, N_q]
        return u, logits


class FineRegressionHead(nn.Module):
    """Direct regression head for fine residual displacement."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_q, d_model] → u: [B, N_q]"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Cross-Attention Block — softmax MHA with RoPE
# ---------------------------------------------------------------------------

class RoPECrossAttentionBlock(nn.Module):
    """Standard softmax multi-head cross-attention with RoPE on Q and K.

    Q: query tokens (from query points)
    K, V: context tokens (from ViT encoder)
    """

    def __init__(self, dim: int = 256, heads: int = 8,
                 dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # RoPE on Q and K
        self.rope = RotaryEmbedding(dim_head, min_freq=1.0 / 256)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor,
                q_coords: torch.Tensor, kv_coords: torch.Tensor):
        """
        Args:
            q_tokens:  [B, N_q, dim]
            kv_tokens: [B, N_kv, dim]
            q_coords:  [B, N_q, 2] normalized coords for RoPE
            kv_coords: [B, N_kv, 2] normalized coords for RoPE

        Returns:
            [B, N_q, dim]
        """
        B, N_q, _ = q_tokens.shape
        N_kv = kv_tokens.shape[1]

        q = self.to_q(q_tokens).view(B, N_q, self.heads, self.dim_head)
        k = self.to_k(kv_tokens).view(B, N_kv, self.heads, self.dim_head)
        v = self.to_v(kv_tokens).view(B, N_kv, self.heads, self.dim_head)

        # Apply RoPE to Q and K
        q = self._apply_rope(q, q_coords)
        k = self._apply_rope(k, kv_coords)

        # Standard scaled dot-product attention
        # [B, heads, N_q, dim_head] @ [B, heads, dim_head, N_kv] → [B, heads, N_q, N_kv]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                     # [B, heads, N_q, dim_head]
        out = out.transpose(1, 2).reshape(B, N_q, -1)   # [B, N_q, inner_dim]
        return self.to_out(out)

    def _apply_rope(self, x, coords):
        """Apply RoPE to a [B, N, heads, dim_head] tensor."""
        B, N, H, D = x.shape
        # RoPE expects [B, N, D] for coordinate encoding
        # Coords shape [B, N, 2]
        # We need to apply RoPE per head
        rope_emb = self.rope(coords)  # [B, N, D]
        # Reshape to match head structure
        sin = rope_emb[:, :, D // 2:]  # [B, N, D//2]
        cos = rope_emb[:, :, :D // 2]  # [B, N, D//2]
        # Apply per head
        x_rot = x.clone()
        for h in range(H):
            x_h = x[:, :, h, :]  # [B, N, D]
            # Rotate half dimensions
            x_pass = x_h[:, :, :D // 2]
            x_rot_h = x_h[:, :, D // 2:]
            x_rot[:, :, h, :D // 2] = x_pass * cos - x_rot_h * sin
            x_rot[:, :, h, D // 2:] = x_rot_h * cos + x_pass * sin
        return x_rot


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """Query-point decoder with dual cross-attention + fusion + dual-scale output.

    Architecture:
        query_points → RoPE → projection → query_tokens
        query_tokens → CrossAttn(ref_tokens) → q_ref
        query_tokens → CrossAttn(tar_tokens) → q_tar
        f_diff = q_tar - q_ref
        fusion = MLP(concat(q_ref, q_tar, f_diff))
        fusion → CoarseHead(x,y) + FineHead(x,y) → u
    """

    def __init__(self, config):
        super().__init__()
        dim = config.feature_dim
        self.dim = dim

        # RoPE for query coordinates
        self.rope = RotaryEmbedding(config.rope_dim, min_freq=config.rope_min_freq)

        # Align RoPE output with feature_dim
        rope_out = config.rope_dim
        if rope_out != dim:
            self.query_proj_in = nn.Linear(rope_out, dim)
        else:
            self.query_proj_in = nn.Identity()

        # Cross-attention blocks (ref and tar branches)
        self.cross_attn_ref = nn.ModuleList([
            RoPECrossAttentionBlock(
                dim=dim, heads=config.attn_heads,
                dim_head=config.attn_dim_head, dropout=config.attn_dropout,
            )
            for _ in range(config.n_cross_attn_layers)
        ])
        self.cross_attn_tar = nn.ModuleList([
            RoPECrossAttentionBlock(
                dim=dim, heads=config.attn_heads,
                dim_head=config.attn_dim_head, dropout=config.attn_dropout,
            )
            for _ in range(config.n_cross_attn_layers)
        ])

        # Layer norms for cross-attention
        self.norm_ref = nn.ModuleList([nn.LayerNorm(dim) for _ in range(config.n_cross_attn_layers)])
        self.norm_tar = nn.ModuleList([nn.LayerNorm(dim) for _ in range(config.n_cross_attn_layers)])

        # Fusion MLP: q_ref + q_tar + f_diff → combined
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 3, config.decoder_mlp_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_mlp_hidden, config.decoder_mlp_out),
            nn.GELU(),
        )

        # Output heads
        self.coarse_head_x = DisplacementHead(config.decoder_mlp_out, config.coarse_bin_centers)
        self.coarse_head_y = DisplacementHead(config.decoder_mlp_out, config.coarse_bin_centers)
        self.fine_head_x = FineRegressionHead(config.decoder_mlp_out)
        self.fine_head_y = FineRegressionHead(config.decoder_mlp_out)

        # Store config for reference
        self.config = config

    def forward(self, query_points, ref_tokens, tar_tokens,
                ref_coords=None, tar_coords=None):
        """
        Args:
            query_points: [B, N_q, 2] normalized coords in [0,1]^2
            ref_tokens:   [B, N_kv, dim] ViT tokens from ref image
            tar_tokens:   [B, N_kv, dim] ViT tokens from tar image
            ref_coords:   [B, N_kv, 2] patch center coords (default: uniform grid)
            tar_coords:   [B, N_kv, 2]

        Returns:
            u_pred:       [B, N_q, 2]
            loss_aux:     dict with coarse/fine logits for loss computation
        """
        B, N_q, _ = query_points.shape

        # Default coordinates for KV tokens (uniform patch grid)
        if ref_coords is None:
            ref_coords = _uniform_patch_coords(
                int(ref_tokens.shape[1] ** 0.5), B, ref_tokens.device,
            )
        if tar_coords is None:
            tar_coords = _uniform_patch_coords(
                int(tar_tokens.shape[1] ** 0.5), B, tar_tokens.device,
            )

        # Encode query points with RoPE
        rope_emb = self.rope(query_points)              # [B, N_q, rope_dim]
        q_tokens = self.query_proj_in(rope_emb)          # [B, N_q, dim]

        # Cross-attend to ref tokens
        q_ref = q_tokens
        for attn, norm in zip(self.cross_attn_ref, self.norm_ref):
            q_ref = q_ref + attn(norm(q_ref), ref_tokens, query_points, ref_coords)

        # Cross-attend to tar tokens
        q_tar = q_tokens
        for attn, norm in zip(self.cross_attn_tar, self.norm_tar):
            q_tar = q_tar + attn(norm(q_tar), tar_tokens, query_points, tar_coords)

        # Feature diff
        f_diff = q_tar - q_ref  # [B, N_q, dim]

        # Fusion
        combined = torch.cat([q_ref, q_tar, f_diff], dim=-1)  # [B, N_q, 3*dim]
        fused = self.fusion_mlp(combined)                      # [B, N_q, out_dim]

        # Dual-scale output
        u_x_coarse, coarse_logits_x = self.coarse_head_x(fused)
        u_y_coarse, coarse_logits_y = self.coarse_head_y(fused)
        u_x_fine = self.fine_head_x(fused)
        u_y_fine = self.fine_head_y(fused)

        u_x = u_x_coarse + u_x_fine
        u_y = u_y_coarse + u_y_fine
        u_pred = torch.stack([u_x, u_y], dim=-1)  # [B, N_q, 2]

        loss_aux = {
            "coarse_logits_x": coarse_logits_x,
            "coarse_logits_y": coarse_logits_y,
            "u_coarse_x": u_x_coarse,
            "u_coarse_y": u_y_coarse,
            "u_fine_x": u_x_fine,
            "u_fine_y": u_y_fine,
        }

        return u_pred, loss_aux


def _uniform_patch_coords(n_patches_per_side: int, batch_size: int,
                          device: torch.device) -> torch.Tensor:
    """Build uniform grid coordinates for ViT patch tokens.

    Returns [B, n_patches, 2] in [0,1]^2.
    """
    n = n_patches_per_side
    ys = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, device=device)
    xs = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx.ravel(), gy.ravel()], dim=-1)  # [n*n, 2]
    return coords.unsqueeze(0).expand(batch_size, -1, -1)
