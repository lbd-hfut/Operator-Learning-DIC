"""PyTorch Dataset for on-the-fly DIC data generation.

Generates synthetic speckle image pairs with known displacement fields
at training time. Supports data augmentation and variable query point counts.
"""
import torch
import numpy as np
from typing import Dict, Optional
from .config import DatasetConfig
from .speckle_generator import SpeckleGenerator
from .image_pool import RealImagePool
from .deformation_generator import DeformationGenerator
from .warp import warp_image
from .noise import add_gaussian_noise
from .sampler import QueryPointSampler
from .augment import DICAugmentation


class DICDataset(torch.utils.data.Dataset):
    """On-the-fly DIC dataset.

    Each __getitem__ generates a new random speckle pattern and deformation,
    warps the reference image, adds noise, samples query points, and returns
    the standardized data dict.

    Args:
        config: DatasetConfig with all generation parameters
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        self.image_pool = None
        if config.mode == "real":
            if config.real_image_dir is None:
                raise ValueError("real_image_dir must be set when mode='real'")
            self.image_pool = RealImagePool(config.real_image_dir)

        self.aug = DICAugmentation(
            image_size=config.image_size,
            h_flip_prob=0.5,
            v_flip_prob=0.5,
            seed=config.seed + 1 if config.seed else None,
        )

    def __len__(self) -> int:
        return self.config.n_samples

    def __getitem__(self, idx: int) -> Dict:
        """Generate a single DIC sample.

        Returns:
            dict with ref_img, tar_img, query_points, u_gt tensors
        """
        # Use idx as sub-seed for per-sample determinism when requested
        sample_seed = self.config.seed + idx + 1000 if self.config.seed else None
        rng = np.random.RandomState(sample_seed)

        cfg = self.config
        H, W = cfg.image_size

        # 1. Generate or load reference image
        if self.image_pool is not None:
            ref_img = self.image_pool.load_random_patch((H, W))
            speckle_size = 0.0
            speckle_density = 0.0
            speckle_contrast = 0.0
        else:
            speckle_size = rng.uniform(*cfg.speckle_size_range)
            speckle_density = rng.uniform(*cfg.speckle_density_range)
            speckle_contrast = rng.uniform(*cfg.speckle_contrast_range)

            speckle_gen = SpeckleGenerator(
                image_size=(H, W),
                particle_size_range=(speckle_size * 0.5, speckle_size),
                density=speckle_density,
                contrast=speckle_contrast,
                seed=sample_seed,
            )
            ref_img = speckle_gen.generate()

        # 2. Generate deformation
        deform_gen = DeformationGenerator(
            image_size=(H, W),
            displacement_range=cfg.displacement_range,
            seed=sample_seed + 1 if sample_seed else None,
        )
        mode = cfg.sample_mode(rng)
        u_field = deform_gen.generate(mode=mode)

        # 3. Warp to create target image
        tar_img = warp_image(ref_img, u_field)

        # 4. Add noise (same noise level for both images)
        noise_std = rng.uniform(*cfg.noise_std_range)
        ref_img = add_gaussian_noise(ref_img, noise_std, seed=sample_seed + 2 if sample_seed else None)
        tar_img = add_gaussian_noise(tar_img, noise_std, seed=sample_seed + 3 if sample_seed else None)

        # 5. Augmentation
        ref_img, tar_img, u_field = self.aug(ref_img, tar_img, u_field)

        # 6. Sample query points
        n_query = rng.randint(cfg.n_query_min, cfg.n_query_max + 1)

        sampler = QueryPointSampler(image_size=(H, W), seed=sample_seed + 4 if sample_seed else None)
        query_pts, u_gt = sampler.sample(
            n_points=n_query,
            u_field=u_field,
            gradient_weighted=cfg.gradient_weighted_sampling,
        )

        # 7. Query point dropout (for generalization)
        if cfg.query_dropout_prob > 0:
            keep_mask = rng.random(n_query) > cfg.query_dropout_prob
            query_pts = query_pts[keep_mask]
            u_gt = u_gt[keep_mask]

        # Convert to tensors
        return {
            "ref_img": torch.from_numpy(ref_img).unsqueeze(0),     # [1, H, W]
            "tar_img": torch.from_numpy(tar_img).unsqueeze(0),     # [1, H, W]
            "query_points": torch.from_numpy(query_pts),            # [N, 2]
            "u_gt": torch.from_numpy(u_gt),                         # [N, 2]
            "deformation_mode": mode,
            "speckle_params": {
                "size": float(speckle_size),
                "density": float(speckle_density),
                "contrast": float(speckle_contrast),
                "noise_std": float(noise_std),
            },
            "sample_id": f"{idx:06d}_{mode}",
        }
