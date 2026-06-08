"""Prediction module for ViT Transformer DIC Operator (Route D).

Usage:
    import dic_vit_method.predict as P

    u = P.predict_dense(ref_img, tar_img, ckpt="checkpoints/route_d/best.pt")

    pred = P.Predictor("checkpoints/route_d/best.pt")
    u_dense = pred.dense(ref_img, tar_img)
    u_sparse = pred.sparse(ref_img, tar_img, pts)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .model import VitDICModel
from .config import VitDICConfig


class Predictor:
    """ViT Transformer DIC predictor (Route D).

    Loads a trained model and exposes dense / sparse inference.

    Parameters
    ----------
    ckpt : str or Path
        Path to a .pt checkpoint containing ``model_state_dict``.
    device : str
        Torch device string (default ``"cuda"`` if available).
    """

    def __init__(self, ckpt: Union[str, Path], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        ckpt = Path(ckpt)
        state = torch.load(ckpt, map_location=self.device, weights_only=False)

        config = state.get("config", VitDICConfig())
        self.model = VitDICModel(config).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dense(
        self,
        ref_img: np.ndarray,
        tar_img: np.ndarray,
    ) -> np.ndarray:
        """Predict full-resolution displacement field.

        Parameters
        ----------
        ref_img : np.ndarray  [H, W] float
            Reference image (grayscale, values in [0, 1]).
        tar_img : np.ndarray  [H, W] float
            Target (deformed) image.

        Returns
        -------
        u : np.ndarray  [H, W, 2]
            Dense predicted displacement in **pixels**.
        """
        H, W = ref_img.shape
        ref_t = _to_tensor(ref_img).to(self.device)
        tar_t = _to_tensor(tar_img).to(self.device)

        ref_tokens, tar_tokens = self.model.encode(ref_t, tar_t)

        # Process in chunks to avoid OOM for 65536 query points
        chunk_size = 4096
        n_total = H * W
        queries = _dense_query_grid(H, W, self.device)  # [1, H*W, 2]

        u_chunks = []
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            q_chunk = queries[:, start:end, :]
            u_chunk = self.model.decode(q_chunk, ref_tokens, tar_tokens)
            u_chunks.append(u_chunk)

        u = torch.cat(u_chunks, dim=1).reshape(H, W, 2).cpu().numpy()
        return u

    @torch.no_grad()
    def sparse(
        self,
        ref_img: np.ndarray,
        tar_img: np.ndarray,
        query_points: np.ndarray,
    ) -> np.ndarray:
        """Predict displacement at arbitrary query points.

        Parameters
        ----------
        ref_img : np.ndarray  [H, W]
        tar_img : np.ndarray  [H, W]
        query_points : np.ndarray  [N, 2]
            Normalised coordinates in [0, 1]^2 (x, y).

        Returns
        -------
        u : np.ndarray  [N, 2]
        """
        ref_t = _to_tensor(ref_img).to(self.device)
        tar_t = _to_tensor(tar_img).to(self.device)
        q_t = torch.from_numpy(query_points.astype(np.float32)).unsqueeze(0).to(self.device)

        ref_tokens, tar_tokens = self.model.encode(ref_t, tar_t)
        u = self.model.decode(q_t, ref_tokens, tar_tokens)
        return u.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Factorised interface (encode once, decode many)
    # ------------------------------------------------------------------

    def encode(self, ref_img: np.ndarray, tar_img: np.ndarray):
        """Return (ref_tokens, tar_tokens) for reuse across query sets."""
        ref_t = _to_tensor(ref_img).to(self.device)
        tar_t = _to_tensor(tar_img).to(self.device)
        return self.model.encode(ref_t, tar_t)

    @torch.no_grad()
    def decode(self, query_points: np.ndarray,
               ref_tokens: torch.Tensor, tar_tokens: torch.Tensor) -> np.ndarray:
        """Decode from pre-computed ViT token sequences."""
        q_t = torch.from_numpy(query_points.astype(np.float32)).unsqueeze(0).to(self.device)
        u = self.model.decode(q_t, ref_tokens, tar_tokens)
        return u.squeeze(0).cpu().numpy()


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def predict_dense(
    ref_img: np.ndarray,
    tar_img: np.ndarray,
    ckpt: Union[str, Path] = "checkpoints/route_d/best.pt",
    device: str = "cuda",
) -> np.ndarray:
    """One-shot dense prediction."""
    return Predictor(ckpt, device).dense(ref_img, tar_img)


def predict_sparse(
    ref_img: np.ndarray,
    tar_img: np.ndarray,
    query_points: np.ndarray,
    ckpt: Union[str, Path] = "checkpoints/route_d/best.pt",
    device: str = "cuda",
) -> np.ndarray:
    """One-shot sparse prediction."""
    return Predictor(ckpt, device).sparse(ref_img, tar_img, query_points)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H, W] → [1, 1, H, W] float32."""
    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def _dense_query_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)
