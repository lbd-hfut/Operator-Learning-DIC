"""Prediction module for U-Net DIC Method (Route C).

Usage:
    import dic_unet_method.predict as P

    u = P.predict_dense(ref_img, tar_img, ckpt="checkpoints/route_c/best.pt")

    pred = P.Predictor("checkpoints/route_c/best.pt")
    u_dense = pred.dense(ref_img, tar_img)           # [H, W, 2]
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from .model import UnetDICModel
from .config import UnetDICConfig


class Predictor:
    """U-Net DIC predictor (Route C).

    Parameters
    ----------
    ckpt : str or Path
        Path to .pt checkpoint.
    device : str
        Torch device (default ``"cuda"`` if available).
    """

    def __init__(self, ckpt: Union[str, Path], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        ckpt = Path(ckpt)
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        config = state.get("config", UnetDICConfig())
        self.model = UnetDICModel(config).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def dense(self, ref_img: np.ndarray, tar_img: np.ndarray) -> np.ndarray:
        """Predict full-resolution displacement field.

        Parameters
        ----------
        ref_img : np.ndarray  [H, W]  float, values in [0, 1]
        tar_img : np.ndarray  [H, W]  float, values in [0, 1]

        Returns
        -------
        u : np.ndarray  [H, W, 2]  displacement in pixels
        """
        ref_t = torch.from_numpy(ref_img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        tar_t = torch.from_numpy(tar_img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        u = self.model(ref_t, tar_t)                        # [1, 2, H, W]
        u = u.squeeze(0).permute(1, 2, 0)                   # [H, W, 2]
        return u.cpu().numpy()


def predict_dense(
    ref_img: np.ndarray,
    tar_img: np.ndarray,
    ckpt: Union[str, Path] = "checkpoints/route_c/best.pt",
    device: str = "cuda",
) -> np.ndarray:
    """One-shot dense prediction."""
    return Predictor(ckpt, device).dense(ref_img, tar_img)
