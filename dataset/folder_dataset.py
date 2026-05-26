"""PyTorch Dataset for pre-generated folder-format DIC data.

Reads from the directory structure produced by generate_dataset_dir:
    split_dir/
      ref/         # reference images as PNG
      tar/         # target images as PNG
      u_field/     # displacement fields as .npy [H, W, 2]
      roi_mask/    # ROI binary masks as PNG
      metadata.csv # per-sample info
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

from .sampler import QueryPointSampler


class FolderDICDataset(torch.utils.data.Dataset):
    """Dataset reading pre-generated DIC data from a folder.

    In each __getitem__, loads ref/tar PNGs, displacement field,
    and ROI mask from disk, then samples query points within the ROI.

    Args:
        split_dir: path to train/validation/test directory
        n_query_min: minimum query points per sample
        n_query_max: maximum query points per sample
        gradient_weighted_sampling: enable gradient-weighted sampling
        normalize_coords: if True, query point coords are [0,1]
        seed: random seed for query point sampling
    """

    def __init__(
        self,
        split_dir: str,
        n_query_min: int = 128,
        n_query_max: int = 2048,
        gradient_weighted_sampling: bool = False,
        normalize_coords: bool = True,
        seed: int = 42,
    ):
        self.split_dir = Path(split_dir)
        self.n_query_min = n_query_min
        self.n_query_max = n_query_max
        self.gradient_weighted_sampling = gradient_weighted_sampling
        self.normalize_coords = normalize_coords

        # Discover samples from ref/ directory
        self.ref_dir = self.split_dir / "ref"
        if not self.ref_dir.is_dir():
            raise FileNotFoundError(f"ref/ not found in {self.split_dir}")

        self._samples = sorted(self.ref_dir.glob("*.png"))
        if not self._samples:
            raise FileNotFoundError(f"No PNG files found in {self.ref_dir}")

        self.n_samples = len(self._samples)

        # Load first u_field to get image dimensions
        u0 = np.load(self.split_dir / "u_field" / f"{0:06d}.npy")
        self.H, self.W = u0.shape[:2]

        self.rng = np.random.RandomState(seed)
        self._sampler = QueryPointSampler(
            image_size=(self.H, self.W), seed=seed + 1
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        # Load images
        ref_img = np.array(Image.open(self.ref_dir / f"{idx:06d}.png"), dtype=np.float32) / 255.0
        tar_img = np.array(Image.open(self.split_dir / "tar" / f"{idx:06d}.png"), dtype=np.float32) / 255.0

        # Load displacement field
        u_field = np.load(self.split_dir / "u_field" / f"{idx:06d}.npy")  # [H, W, 2]

        # Load ROI mask
        roi_path = self.split_dir / "roi_mask" / f"{idx:06d}.png"
        if roi_path.exists():
            roi_mask = np.array(Image.open(roi_path)) > 127
        else:
            roi_mask = np.ones((self.H, self.W), dtype=bool)

        # Sample query points within ROI
        n_query = self.rng.randint(self.n_query_min, self.n_query_max + 1)
        query_pts, u_gt = self._sampler.sample(
            n_points=n_query,
            u_field=u_field,
            roi_mask=roi_mask,
            gradient_weighted=self.gradient_weighted_sampling,
        )

        # Filter out any points that might still be outside ROI
        if roi_mask is not None:
            px = np.round(query_pts[:, 0] * (self.W - 1)).astype(int)
            py = np.round(query_pts[:, 1] * (self.H - 1)).astype(int)
            px = np.clip(px, 0, self.W - 1)
            py = np.clip(py, 0, self.H - 1)
            keep = roi_mask[py, px]
            query_pts = query_pts[keep]
            u_gt = u_gt[keep]

        return {
            "ref_img": torch.from_numpy(ref_img).unsqueeze(0),   # [1, H, W]
            "tar_img": torch.from_numpy(tar_img).unsqueeze(0),   # [1, H, W]
            "query_points": torch.from_numpy(query_pts),          # [N, 2]
            "u_gt": torch.from_numpy(u_gt),                       # [N, 2]
            "sample_id": f"{idx:06d}",
        }
