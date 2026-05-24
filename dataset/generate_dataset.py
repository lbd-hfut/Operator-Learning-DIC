"""CLI script to pre-generate DIC dataset and cache to HDF5.

Usage:
    python -m dataset.generate_dataset --n_samples 1000 --output data/train.h5
    python -m dataset.generate_dataset --config my_config.yaml --output data/train.h5
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .config import DatasetConfig
from .speckle_generator import SpeckleGenerator
from .deformation_generator import DeformationGenerator
from .warp import warp_image
from .noise import add_gaussian_noise


def generate_dataset(config: DatasetConfig, output_path: str):
    """Generate and save dataset to HDF5.

    Args:
        config: DatasetConfig specifying all generation parameters
        output_path: path to output HDF5 file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = config.image_size
    N = config.n_samples
    rng = np.random.RandomState(config.seed)

    with h5py.File(output_path, "w") as f:
        # Create datasets
        img_ref = f.create_dataset(
            "images/ref", (N, H, W), dtype="float32",
            chunks=(1, H, W), compression="gzip", compression_opts=4,
        )
        img_tar = f.create_dataset(
            "images/tar", (N, H, W), dtype="float32",
            chunks=(1, H, W), compression="gzip", compression_opts=4,
        )
        u_fields = f.create_dataset(
            "u_fields", (N, H, W, 2), dtype="float32",
            chunks=(1, H, W, 2), compression="gzip", compression_opts=4,
        )
        dt = h5py.string_dtype()
        modes = f.create_dataset("deformation_modes", (N,), dtype=dt)

        for i in tqdm(range(N), desc="Generating samples"):
            sample_seed = config.seed + i if config.seed else None

            # Speckle
            speckle_size = rng.uniform(*config.speckle_size_range)
            speckle_density = rng.uniform(*config.speckle_density_range)
            speckle_contrast = rng.uniform(*config.speckle_contrast_range)
            sg = SpeckleGenerator(
                image_size=(H, W),
                particle_size_range=(speckle_size * 0.5, speckle_size),
                density=speckle_density,
                contrast=speckle_contrast,
                seed=sample_seed,
            )
            ref = sg.generate()

            # Deformation
            dg = DeformationGenerator(
                image_size=(H, W),
                displacement_range=config.displacement_range,
                seed=sample_seed + 1 if sample_seed else None,
            )
            mode = config.deformation_modes[rng.randint(0, len(config.deformation_modes))]
            u_field = dg.generate(mode=mode)

            # Warp
            tar = warp_image(ref, u_field)

            # Noise
            noise_std = rng.uniform(*config.noise_std_range)
            ref = add_gaussian_noise(ref, noise_std, seed=sample_seed + 2 if sample_seed else None)
            tar = add_gaussian_noise(tar, noise_std, seed=sample_seed + 3 if sample_seed else None)

            img_ref[i] = ref
            img_tar[i] = tar
            u_fields[i] = u_field
            modes[i] = mode

    print(f"Dataset saved to {output_path} ({N} samples)")


def main():
    parser = argparse.ArgumentParser(description="Generate DIC dataset")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/dic_train.h5")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = DatasetConfig(
        n_samples=args.n_samples,
        image_size=tuple(args.image_size),
        seed=args.seed,
    )
    generate_dataset(config, args.output)


if __name__ == "__main__":
    main()
