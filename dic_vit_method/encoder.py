"""ViT Encoder for Route D.

Wraps a frozen torchvision ViT-B/16 to produce patch tokens, then
projects to feature_dim and adds RoPE position encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from common.rotary_embedding import RotaryEmbedding


class ViTEncoder(nn.Module):
    """Frozen ViT + learnable projection + RoPE.

    Input:  [B, 1, H, W] grayscale image
    Output: [B, n_patches, feature_dim] token sequence
    """

    def __init__(
        self,
        feature_dim: int = 256,
        vit_feature_dim: int = 768,
        n_patches: int = 256,
        rope_dim: int = 256,
        rope_min_freq: float = 1.0 / 256,
        freeze: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.vit_feature_dim = vit_feature_dim
        self.n_patches = n_patches

        # Load pre-trained ViT with 256×256 input.
        # torchvision enforces image_size=224 for IMAGENET1K weights,
        # so we create with image_size=256 without weights, then load
        # state dict and interpolate position embeddings manually.
        self.vit = vit_b_16(weights=None, image_size=256)
        self.patch_size = self.vit.patch_size  # 16

        if pretrained:
            state = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).state_dict()
            # Interpolate position embedding from 197 (14²+1) to 257 (16²+1)
            old_pos = state["encoder.pos_embedding"]  # [1, 197, 768]
            old_n = int((old_pos.shape[1] - 1) ** 0.5)  # 14
            new_n = 256 // self.patch_size                # 16
            cls_emb = old_pos[:, :1, :]                   # [1, 1, 768]
            patch_emb = old_pos[:, 1:, :]                 # [1, 196, 768]
            patch_emb = patch_emb.reshape(1, old_n, old_n, -1).permute(0, 3, 1, 2)
            patch_emb = F.interpolate(patch_emb, size=(new_n, new_n),
                                      mode="bicubic", align_corners=False)
            patch_emb = patch_emb.permute(0, 2, 3, 1).reshape(1, new_n * new_n, -1)
            state["encoder.pos_embedding"] = torch.cat([cls_emb, patch_emb], dim=1)
            self.vit.load_state_dict(state)

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        # Project ViT tokens to feature_dim
        self.proj = nn.Linear(vit_feature_dim, feature_dim)

        # RoPE for patch position encoding
        H_patches = W_patches = int(n_patches ** 0.5)  # 16 for 256/16
        patch_centers = self._build_patch_coords(H_patches, W_patches)  # [n_patches, 2]
        self.register_buffer("patch_coords", patch_centers)
        self.rope = RotaryEmbedding(rope_dim, min_freq=rope_min_freq)

    @staticmethod
    def _build_patch_coords(h: int, w: int):
        """Build normalized coordinates for patch centers.

        Returns [h*w, 2] in [0, 1]^2 (x, y order).
        """
        ys = torch.linspace(0.5 / h, 1.0 - 0.5 / h, h)
        xs = torch.linspace(0.5 / w, 1.0 - 0.5 / w, w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.ravel(), gy.ravel()], dim=-1)  # [N, 2]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a grayscale image to token sequence.

        Args:
            img: [B, 1, H, W] grayscale image, values in [0, 1]

        Returns:
            tokens: [B, n_patches, feature_dim]
        """
        B = img.shape[0]

        # Convert grayscale to RGB by repeating channels
        if img.shape[1] == 1:
            img_rgb = img.repeat(1, 3, 1, 1)
        else:
            img_rgb = img

        # ViT forward — extract patch tokens from encoder
        with torch.set_grad_enabled(not all(p.requires_grad for p in self.vit.parameters())):
            # Patch embedding
            x = self.vit.conv_proj(img_rgb)                # [B, C, H/p, W/p]
            x = x.flatten(2).transpose(1, 2)               # [B, n_patches, C]
            # Prepend CLS token
            n_p = x.shape[0]
            cls_token = self.vit.class_token.expand(n_p, -1, -1)
            x = torch.cat([cls_token, x], dim=1)           # [B, 1 + n_patches, C]
            # Encoder (adds pos_embedding internally)
            x = self.vit.encoder(x)                         # [B, n_patches+1, C]

        # Remove CLS token → keep only patch tokens
        tokens = x[:, 1:, :]  # [B, n_patches, C]

        # Project to feature_dim
        tokens = self.proj(tokens)  # [B, n_patches, feature_dim]

        # Add RoPE position encoding
        pos_coords = self.patch_coords.unsqueeze(0).expand(B, -1, -1)  # [B, n_patches, 2]
        pos_emb = self.rope(pos_coords)  # [B, n_patches, rope_dim]

        # Align RoPE output with feature_dim
        if pos_emb.shape[-1] != self.feature_dim:
            # RoPE output is rope_dim, we need to handle the mismatch
            # Truncate or repeat to match feature_dim
            pos_emb = F.pad(pos_emb, (0, self.feature_dim - pos_emb.shape[-1]))

        tokens = tokens + pos_emb

        return tokens
