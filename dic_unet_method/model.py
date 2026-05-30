"""U-Net model for dense DIC displacement prediction (Route C).

Takes concatenated [I_ref, I_tar] and directly outputs displacement
field u [B, 2, H, W].  Standard encoder-decoder with skip connections.

Unlike Route A/B (query-point operators), this is a dense image-to-image
model — one forward pass predicts the full displacement field.
"""
import torch
import torch.nn as nn

from .config import UnetDICConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv-Norm-Act block used in the U-Net encoder/decoder."""

    def __init__(self, in_ch: int, out_ch: int, use_group_norm: bool = True):
        super().__init__()
        norm = nn.GroupNorm(8, out_ch) if use_group_norm else nn.BatchNorm2d(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm,
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two ConvBlock layers in sequence."""

    def __init__(self, in_ch: int, out_ch: int, use_group_norm: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, out_ch, use_group_norm),
            ConvBlock(out_ch, out_ch, use_group_norm),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UnetDICModel(nn.Module):
    """U-Net for dense DIC displacement prediction.

    Architecture:
      Encoder: 4× max-pool + DoubleConv stages
      Bottleneck: DoubleConv at coarsest resolution
      Decoder: 4× ConvTranspose2d upsampling + skip connections
      Head: Conv2d → u [B, 2, H, W]

    Parameters
    ----------
    config : UnetDICConfig
    """

    def __init__(self, config: UnetDICConfig):
        super().__init__()
        self.config = config

        ch = config.base_channels
        mults = config.channel_multipliers
        gn = config.use_group_norm

        # ---- Encoder ----
        self.enc1 = DoubleConv(config.in_channels, ch * mults[0], gn)           # 256
        self.pool1 = nn.MaxPool2d(2)                                            # 128
        self.enc2 = DoubleConv(ch * mults[0], ch * mults[1], gn)
        self.pool2 = nn.MaxPool2d(2)                                            # 64
        self.enc3 = DoubleConv(ch * mults[1], ch * mults[2], gn)
        self.pool3 = nn.MaxPool2d(2)                                            # 32
        self.enc4 = DoubleConv(ch * mults[2], ch * mults[3], gn)
        self.pool4 = nn.MaxPool2d(2)                                            # 16

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(ch * mults[3], ch * mults[4], gn)          # 16

        # ---- Decoder ----
        self.up4 = nn.ConvTranspose2d(ch * mults[4], ch * mults[3], 2, stride=2)
        self.dec4 = DoubleConv(ch * mults[3] * 2, ch * mults[3], gn)

        self.up3 = nn.ConvTranspose2d(ch * mults[3], ch * mults[2], 2, stride=2)
        self.dec3 = DoubleConv(ch * mults[2] * 2, ch * mults[2], gn)

        self.up2 = nn.ConvTranspose2d(ch * mults[2], ch * mults[1], 2, stride=2)
        self.dec2 = DoubleConv(ch * mults[1] * 2, ch * mults[1], gn)

        self.up1 = nn.ConvTranspose2d(ch * mults[1], ch * mults[0], 2, stride=2)
        self.dec1 = DoubleConv(ch * mults[0] * 2, ch * mults[0], gn)

        # ---- Output head ----
        self.head = nn.Conv2d(ch * mults[0], config.out_channels, 3, padding=1)
        nn.init.constant_(self.head.bias, 0.0)

    # ------------------------------------------------------------------
    # Public API  (compatible with Route A / B call patterns)
    # ------------------------------------------------------------------

    def forward(
        self,
        ref_img: torch.Tensor,
        tar_img: torch.Tensor,
        query_points: torch.Tensor = None,   # ignored — kept for API compatibility
    ) -> torch.Tensor:
        """Predict dense displacement field.

        Parameters
        ----------
        ref_img : [B, 1, H, W]
        tar_img : [B, 1, H, W]
        query_points : ignored (Route C is dense, no query points needed)

        Returns
        -------
        u : [B, 2, H, W]  displacement in pixels
        """
        x = torch.cat([ref_img, tar_img], dim=1)  # [B, 2, H, W]

        # Encoder
        e1 = self.enc1(x)                           # 256
        e2 = self.enc2(self.pool1(e1))              # 128
        e3 = self.enc3(self.pool2(e2))              # 64
        e4 = self.enc4(self.pool3(e3))              # 32

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))          # 16

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
