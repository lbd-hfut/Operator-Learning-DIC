"""Prediction module for DIC Solver Operator (Route A).

Usage:
    import dic_solver_operator.predict as P

    # Quick predict from image arrays
    u = P.predict_dense(ref_img, tar_img, ckpt="checkpoints/route_a/best.pt")

    # Or use the Predictor class for repeated inference
    pred = P.Predictor("checkpoints/route_a/best.pt")
    u_dense = pred.dense(ref_img, tar_img)           # full [H, W, 2] field
    u_sparse = pred.sparse(ref_img, tar_img, pts)    # [N, 2] at query points
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .model import SolverOperatorModel
from .config import SolverOperatorConfig


class Predictor:
    """DIC Solver Operator predictor (Route A).

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

        config = state.get("config", SolverOperatorConfig())
        self.model = SolverOperatorModel(config).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        self._feature_dim = config.feature_dim

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

        f_enc = self.model.encode(ref_t, tar_t)
        queries = _dense_query_grid(H, W, self.device)
        u = self.model.decode(queries, f_enc)
        return u.reshape(H, W, 2).cpu().numpy()

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
            Normalised coordinates in [0, 1]² (x, y).

        Returns
        -------
        u : np.ndarray  [N, 2]
        """
        ref_t = _to_tensor(ref_img).to(self.device)
        tar_t = _to_tensor(tar_img).to(self.device)
        q_t = torch.from_numpy(query_points.astype(np.float32)).unsqueeze(0).to(self.device)

        f_enc = self.model.encode(ref_t, tar_t)
        u = self.model.decode(q_t, f_enc)
        return u.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Convenience: dict-style factorised interface
    # ------------------------------------------------------------------

    def encode(self, ref_img: np.ndarray, tar_img: np.ndarray) -> torch.Tensor:
        """Return the encoded feature field (for reuse across query sets)."""
        ref_t = _to_tensor(ref_img).to(self.device)
        tar_t = _to_tensor(tar_img).to(self.device)
        return self.model.encode(ref_t, tar_t)

    @torch.no_grad()
    def decode(self, query_points: np.ndarray, f_enc: torch.Tensor) -> np.ndarray:
        """Decode from a pre-computed feature field."""
        q_t = torch.from_numpy(query_points.astype(np.float32)).unsqueeze(0).to(self.device)
        u = self.model.decode(q_t, f_enc)
        return u.squeeze(0).cpu().numpy()


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def predict_dense(
    ref_img: np.ndarray,
    tar_img: np.ndarray,
    ckpt: Union[str, Path] = "checkpoints/route_a/best.pt",
    device: str = "cuda",
) -> np.ndarray:
    """One-shot dense prediction. Loads model on every call — use Predictor for batches."""
    return Predictor(ckpt, device).dense(ref_img, tar_img)


def predict_sparse(
    ref_img: np.ndarray,
    tar_img: np.ndarray,
    query_points: np.ndarray,
    ckpt: Union[str, Path] = "checkpoints/route_a/best.pt",
    device: str = "cuda",
) -> np.ndarray:
    """One-shot sparse prediction."""
    return Predictor(ckpt, device).sparse(ref_img, tar_img, query_points)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H, W] -> [1, 1, H, W] float32."""
    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def _dense_query_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)
