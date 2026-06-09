"""Hybrid CNN+ViT Encoder for Route D.

Uses a lightweight CNN stem to extract fine-grained speckle features
(256×256 → 16×16 tokens), then passes through frozen ViT blocks for
global feature enhancement. RoPE encodes spatial position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from common.rotary_embedding import RotaryEmbedding


# ---------------------------------------------------------------------------
# Lightweight ResBlock (same pattern as Route A/B)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual conv block: Conv-GN-GELU → Conv-GN → +residual → GELU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        r = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)


# ---------------------------------------------------------------------------
# CNN Stem — grayscale → feature map
# ---------------------------------------------------------------------------

class CNNStem(nn.Module):
    """Lightweight CNN: 256×256 grayscale → 16×16 feature tokens.

    4 stages, each stride-2, with ResBlocks for local feature extraction.
    Output channels approximately doubling each stage.
    """

    def __init__(self, channels=(64, 96, 128, 192), out_dim=768):
        super().__init__()
        self.out_dim = out_dim

        # Stem conv
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 7, stride=1, padding=3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
        )
        current_ch = channels[0]

        # 4 stages: 256 → 128 → 64 → 32 → 16
        self.stages = nn.ModuleList()
        for ch in channels:
            stride = 2
            self.stages.append(ResBlock(current_ch, ch, stride=stride))
            current_ch = ch

        # Project to ViT dimension
        self.proj = nn.Conv2d(current_ch, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 256, 256]

        Returns:
            [B, 256, out_dim] token sequence
        """
        x = self.stem(x)                    # [B, 64, 256, 256]
        for stage in self.stages:
            x = stage(x)                    # [B, 64→128, 128→64→32→16, °]
        x = self.proj(x)                    # [B, out_dim, 16, 16]
        x = x.flatten(2).transpose(1, 2)   # [B, 256, out_dim]
        return x


# ---------------------------------------------------------------------------
# Hybrid CNN+ViT Encoder
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """Hybrid CNN+ViT encoder: CNN stem → frozen ViT blocks → projected tokens.

    Input:  [B, 1, H, W] grayscale
    Output: [B, n_tokens, feature_dim] token sequence
    """

    def __init__(
        self,
        feature_dim: int = 256,
        vit_feature_dim: int = 768,
        n_patches: int = 256,
        rope_dim: int = 256,
        rope_min_freq: float = 1.0 / 256,
        freeze_vit: bool = True,
        pretrained: bool = True,
        cnn_channels=(64, 96, 128, 192),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_patches = n_patches

        # ---- CNN stem (trainable) ----
        self.cnn_stem = CNNStem(channels=cnn_channels, out_dim=vit_feature_dim)

        # ---- ViT encoder blocks (frozen) ----
        # Load ViT to get encoder blocks + position embedding
        self.vit = vit_b_16(weights=None, image_size=256)
        if pretrained:
            state = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).state_dict()
            old_pos = state["encoder.pos_embedding"]       # [1, 197, 768]
            old_n = int((old_pos.shape[1] - 1) ** 0.5)     # 14
            new_n = int(n_patches ** 0.5)                   # 16
            cls_emb = old_pos[:, :1, :]
            patch_emb = old_pos[:, 1:, :].reshape(1, old_n, old_n, -1).permute(0, 3, 1, 2)
            patch_emb = F.interpolate(patch_emb, size=(new_n, new_n),
                                      mode="bicubic", align_corners=False)
            patch_emb = patch_emb.permute(0, 2, 3, 1).reshape(1, new_n * new_n, -1)
            state["encoder.pos_embedding"] = torch.cat([cls_emb, patch_emb], dim=1)
            self.vit.load_state_dict(state, strict=False)

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

        # ---- Project ViT tokens to decoder dimension ----
        self.proj = nn.Linear(vit_feature_dim, feature_dim)

        # ---- RoPE token position encoding ----
        grid_n = int(n_patches ** 0.5)
        self.register_buffer("patch_coords", self._build_coords(grid_n, grid_n))
        self.rope = RotaryEmbedding(rope_dim, min_freq=rope_min_freq)

    @staticmethod
    def _build_coords(h, w):
        ys = torch.linspace(0.5 / h, 1.0 - 0.5 / h, h)
        xs = torch.linspace(0.5 / w, 1.0 - 0.5 / w, w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.ravel(), gy.ravel()], dim=-1)  # [N, 2]

    def forward(self, img):
        """
        Args:
            img: [B, 1, H, W] grayscale image in [0, 1]

        Returns:
            tokens: [B, n_patches, feature_dim]
        """
        B = img.shape[0]

        # ---- CNN stem (trainable) ----
        tokens = self.cnn_stem(img)  # [B, 256, 768]

        # ---- ViT encoder blocks (frozen) ----
        with torch.set_grad_enabled(not all(p.requires_grad for p in self.vit.parameters())):
            # Prepend CLS token
            cls = self.vit.class_token.expand(B, -1, -1)
            x = torch.cat([cls, tokens], dim=1)            # [B, 257, 768]
            x = self.vit.encoder(x)                         # [B, 257, 768]

        # Drop CLS
        tokens = x[:, 1:, :]  # [B, 256, 768]

        # ---- Project to decoder dim ----
        tokens = self.proj(tokens)  # [B, 256, feature_dim]

        # ---- Add RoPE ----
        coords = self.patch_coords.unsqueeze(0).expand(B, -1, -1)
        pos = self.rope(coords)
        if pos.shape[-1] < self.feature_dim:
            pos = F.pad(pos, (0, self.feature_dim - pos.shape[-1]))
        tokens = tokens + pos

        return tokens
