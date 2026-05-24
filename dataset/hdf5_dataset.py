"""PyTorch Dataset for pre-generated HDF5 DIC data.

Reads from HDF5 files for efficient, reproducible data loading
without on-the-fly generation overhead.
"""
import torch
import h5py
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class HDF5DICDataset(torch.utils.data.Dataset):
    """Dataset reading pre-generated DIC data from HDF5.

    HDF5 file structure:
        /images/ref       [N, H, W] float32
        /images/tar       [N, H, W] float32
        /u_fields         [N, H, W, 2] float32  (full-field GT)
        /deformation_modes [N] string
        /speckle_params   [N] compound or group

    During __getitem__, query points are sampled from the stored
    full-field displacement, so query count and distribution
    remain flexible at training time.

    Args:
        hdf5_path: path to HDF5 file
        n_query_min: minimum query points per sample
        n_query_max: maximum query points per sample
        gradient_weighted_sampling: enable gradient-weighted sampling
        seed: random seed for sampling
    """

    def __init__(
        self,
        hdf5_path: str,
        n_query_min: int = 128,
        n_query_max: int = 2048,
        gradient_weighted_sampling: bool = False,
        seed: int = 42,
    ):
        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.h5_file = h5py.File(hdf5_path, "r")
        self.n_samples = self.h5_file["images/ref"].shape[0]
        self.H = self.h5_file["images/ref"].shape[1]
        self.W = self.h5_file["images/ref"].shape[2]

        self.n_query_min = n_query_min
        self.n_query_max = n_query_max
        self.gradient_weighted_sampling = gradient_weighted_sampling
        self.rng = np.random.RandomState(seed)

        from .sampler import QueryPointSampler
        self._sampler = QueryPointSampler(image_size=(self.H, self.W), seed=seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        ref_img = self.h5_file["images/ref"][idx]
        tar_img = self.h5_file["images/tar"][idx]
        u_field = self.h5_file["u_fields"][idx]

        n_query = self.rng.randint(self.n_query_min, self.n_query_max + 1)

        query_pts, u_gt = self._sampler.sample(
            n_points=n_query,
            u_field=u_field,
            gradient_weighted=self.gradient_weighted_sampling,
        )

        mode = self.h5_file["deformation_modes"][idx]
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")

        return {
            "ref_img": torch.from_numpy(ref_img).unsqueeze(0),
            "tar_img": torch.from_numpy(tar_img).unsqueeze(0),
            "query_points": torch.from_numpy(query_pts),
            "u_gt": torch.from_numpy(u_gt),
            "deformation_mode": mode,
            "sample_id": f"{idx:06d}_{mode}",
        }

    def close(self):
        self.h5_file.close()

    def __del__(self):
        if hasattr(self, "h5_file"):
            self.h5_file.close()
