"""CLI script to pre-generate DIC dataset and cache to HDF5 or folder.

Usage:
    # Default: real images, folder output, auto timestamp path, train only
    python -m dataset.generate_dataset --train 100

    # Train/val/test split with folder output
    python -m dataset.generate_dataset --train 800 --val 100 --test 100

    # HDF5 output for production
    python -m dataset.generate_dataset --train 5000 --output_format h5

    # Custom output path (overrides auto timestamp)
    python -m dataset.generate_dataset --train 1000 --output my_dataset/

    # Synthetic speckle (legacy)
    python -m dataset.generate_dataset --mode synthetic --n_samples 1000
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import DatasetConfig
from .speckle_generator import SpeckleGenerator
from .image_pool import RealImagePool
from .deformation_generator import DeformationGenerator
from .warp import warp_image
from .noise import add_gaussian_noise
from .roi import compute_valid_mask


def generate_dataset(config: DatasetConfig, output_path: str):
    """Generate and save dataset to HDF5.

    Args:
        config: DatasetConfig specifying all generation parameters
        output_path: path to output HDF5 file
    """
    import h5py

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = config.image_size
    N = config.n_samples
    rng = np.random.RandomState(config.seed)

    # Real image pool (only for mode="real")
    image_pool = None
    if config.mode == "real":
        if config.real_image_dir is None:
            raise ValueError("real_image_dir must be set when mode='real'")
        image_pool = RealImagePool(config.real_image_dir)

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
        roi_masks = f.create_dataset(
            "roi_masks", (N, H, W), dtype="bool",
            chunks=(1, H, W), compression="gzip", compression_opts=4,
        )
        dt = h5py.string_dtype()
        modes = f.create_dataset("deformation_modes", (N,), dtype=dt)

        for i in tqdm(range(N), desc="Generating samples"):
            sample_seed = config.seed + i if config.seed else None

            # Reference image: synthetic speckle or real image crop
            if image_pool is not None:
                ref = image_pool.load_random_patch((H, W))
            else:
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
            mode = config.sample_mode(rng)
            u_field = dg.generate(mode=mode)

            # Warp
            tar = warp_image(ref, u_field)

            # Noise
            noise_std = rng.uniform(*config.noise_std_range)
            ref = add_gaussian_noise(ref, noise_std, seed=sample_seed + 2 if sample_seed else None)
            tar = add_gaussian_noise(tar, noise_std, seed=sample_seed + 3 if sample_seed else None)

            # ROI
            valid_mask = compute_valid_mask(u_field)

            img_ref[i] = ref
            img_tar[i] = tar
            u_fields[i] = u_field
            roi_masks[i] = valid_mask
            modes[i] = mode

    print(f"Dataset saved to {output_path} ({N} samples)")


def generate_dataset_dir(config: DatasetConfig, output_dir: str):
    """Generate and save dataset as individual image files.

    Directory structure:
        output_dir/
          ref/       # reference images as PNG
          tar/       # target (warped) images as PNG
          u_field/   # displacement fields as .npy
          metadata.csv  # index, deformation_mode, noise_std, etc.
    """
    output_dir = Path(output_dir)
    (output_dir / "ref").mkdir(parents=True, exist_ok=True)
    (output_dir / "tar").mkdir(parents=True, exist_ok=True)
    (output_dir / "u_field").mkdir(parents=True, exist_ok=True)
    (output_dir / "roi_mask").mkdir(parents=True, exist_ok=True)

    H, W = config.image_size
    N = config.n_samples
    rng = np.random.RandomState(config.seed)

    image_pool = None
    if config.mode == "real":
        if config.real_image_dir is None:
            raise ValueError("real_image_dir must be set when mode='real'")
        image_pool = RealImagePool(config.real_image_dir)

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "deformation_mode", "noise_std", "roi_coverage"])

        for i in tqdm(range(N), desc="Generating samples"):
            sample_seed = config.seed + i if config.seed else None

            # Reference image
            if image_pool is not None:
                ref = image_pool.load_random_patch((H, W))
            else:
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
            mode = config.sample_mode(rng)
            u_field = dg.generate(mode=mode)

            # Warp
            tar = warp_image(ref, u_field)

            # Noise
            noise_std = rng.uniform(*config.noise_std_range)
            ref = add_gaussian_noise(ref, noise_std, seed=sample_seed + 2 if sample_seed else None)
            tar = add_gaussian_noise(tar, noise_std, seed=sample_seed + 3 if sample_seed else None)

            # ROI: per-pixel valid mask from x + u(x) within image bounds
            valid_mask = compute_valid_mask(u_field)
            coverage = valid_mask.sum() / valid_mask.size

            # Save
            ref_img_pil = Image.fromarray((ref * 255).clip(0, 255).astype(np.uint8))
            tar_img_pil = Image.fromarray((tar * 255).clip(0, 255).astype(np.uint8))
            ref_img_pil.save(output_dir / "ref" / f"{i:06d}.png")
            tar_img_pil.save(output_dir / "tar" / f"{i:06d}.png")
            np.save(output_dir / "u_field" / f"{i:06d}.npy", u_field)
            roi_pil = Image.fromarray((valid_mask * 255).astype(np.uint8))
            roi_pil.save(output_dir / "roi_mask" / f"{i:06d}.png")
            writer.writerow([f"{i:06d}", mode, f"{noise_std:.4f}",
                              f"{coverage:.4f}"])

    print(f"Dataset saved to {output_dir} ({N} samples)")


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load dataset generation config from a YAML file.

    Returns a dict of overrides to pass to DatasetConfig.
    """
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate DIC dataset")
    parser.add_argument("--config", type=str, default=None,
                        help="path to YAML config file (overrides CLI defaults)")
    parser.add_argument("--train", type=int, default=0,
                        help="number of training samples")
    parser.add_argument("--val", type=int, default=0,
                        help="number of validation samples")
    parser.add_argument("--test", type=int, default=0,
                        help="number of test samples")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="legacy: train-only sample count (ignored if --train is set)")
    parser.add_argument("--output", type=str, default=None,
                        help="output root dir (default: dataset/dataset/<YYYY-MM-DD>/)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="real",
                        choices=["synthetic", "real"])
    parser.add_argument("--real_image_dir", type=str,
                        default="dataset/original_image")
    parser.add_argument("--output_format", type=str, default="dir",
                        choices=["h5", "dir"],
                        help="dir: individual PNG+NPY files; h5: single HDF5 file")
    args = parser.parse_args()

    # Load YAML config as base, CLI args override
    yaml_cfg = {}
    if args.config:
        yaml_cfg = load_config_from_yaml(args.config)
        print(f"Loaded config from {args.config}")

    # Resolve relative paths from project root
    project_root = Path(__file__).resolve().parents[1]
    if not Path(args.real_image_dir).is_absolute() and args.mode == "real":
        args.real_image_dir = str(project_root / args.real_image_dir)
    if args.output and not Path(args.output).is_absolute():
        args.output = str(project_root / args.output)

    # Build split plan: YAML base, CLI overrides
    yaml_splits = yaml_cfg.get("splits", {})
    splits = {}
    train_n = args.train or yaml_splits.get("train", 0)
    val_n = args.val or yaml_splits.get("validation", 0)
    test_n = args.test or yaml_splits.get("test", 0)
    if train_n > 0 or val_n > 0 or test_n > 0:
        if train_n > 0:
            splits["train"] = train_n
        if val_n > 0:
            splits["validation"] = val_n
        if test_n > 0:
            splits["test"] = test_n
    else:
        splits["train"] = args.n_samples

    # Merge deformation mode weights from YAML
    yaml_modes = yaml_cfg.get("deformation_modes", {})
    # Get default mode weights from dataclass field
    default_modes = DatasetConfig.__dataclass_fields__["deformation_modes"].default_factory()
    mode_weights = dict(default_modes)
    mode_weights.update(yaml_modes)

    # Auto output path
    if args.output is None:
        from datetime import date
        timestamp = date.today().strftime("%Y-%m-%d")
        args.output = f"dataset/dataset/{timestamp}"

    print(f"Output root: {args.output}")
    print(f"Splits: {splits}")

    base_seed = args.seed
    image_size = tuple(args.image_size)

    for split_name, n_samples in splits.items():
        config = DatasetConfig(
            n_samples=n_samples,
            image_size=image_size,
            seed=base_seed,
            mode=args.mode,
            real_image_dir=args.real_image_dir,
            deformation_modes=mode_weights,
            displacement_range=tuple(yaml_cfg.get("displacement_range", [0.1, 20.0])),
            noise_std_range=tuple(yaml_cfg.get("noise_std_range", [0.0, 0.03])),
        )

        if args.output_format == "dir":
            split_dir = f"{args.output}/{split_name}"
            generate_dataset_dir(config, split_dir)
        else:
            split_path = f"{args.output}/{split_name}.h5"
            generate_dataset(config, split_path)

        # Use different seed per split for diversity
        base_seed += 10000


if __name__ == "__main__":
    main()
