"""Evaluation script for Deformation Inverse Operator (Route B).

Loads a trained checkpoint and evaluates on a test dataset.
Includes latent space visualization (PCA/t-SNE of z).

Usage:
    python -m deformation_inverse_operator.eval --checkpoint path/to/checkpoint.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.config import DatasetConfig
from dataset.dic_dataset import DICDataset
from dataset.collate import collate_fn
from deformation_inverse_operator.config import InverseOperatorConfig
from deformation_inverse_operator.model import InverseOperatorModel
from common.metrics import relative_l2_error, angular_error
from common.checkpoint import load_checkpoint


def evaluate(model, dataloader, device):
    """Run evaluation on test set."""
    model.eval()
    all_rel_l2 = []
    all_angle_err = []
    latent_codes = []
    deformation_modes = []

    with torch.no_grad():
        for batch in dataloader:
            ref = batch["ref_img"].to(device)
            tar = batch["tar_img"].to(device)
            qpts = batch["query_points"].to(device)
            u_gt = batch["u_gt"].to(device)
            qmask = batch["query_mask"].to(device)

            # Get latent code and predictions
            z = model.encode(ref, tar)
            u_pred = model.decode(qpts, z)

            # Save latent codes for visualization
            latent_codes.append(z.cpu().numpy())
            deformation_modes.extend(batch.get("deformation_mode", ["unknown"] * ref.shape[0]))

            # Per-sample metrics
            for b in range(ref.shape[0]):
                m = qmask[b]
                if m.sum() == 0:
                    continue
                rel_l2 = relative_l2_error(u_pred[b:b+1], u_gt[b:b+1], m.unsqueeze(0))
                ang_err = angular_error(u_pred[b:b+1], u_gt[b:b+1], m.unsqueeze(0))
                all_rel_l2.append(rel_l2.item())
                all_angle_err.append(ang_err.item())

        latent_codes = np.concatenate(latent_codes, axis=0)  # [N_samples, M, d]

    print(f"=== Evaluation Results (Route B) ===")
    print(f"Samples evaluated: {len(all_rel_l2)}")
    print(f"Relative L2 Error:  {np.mean(all_rel_l2):.4f} ± {np.std(all_rel_l2):.4f}")
    print(f"Angular Error (deg): {np.mean(all_angle_err):.2f} ± {np.std(all_angle_err):.2f}")
    print(f"Latent code shape:   {latent_codes.shape}")

    return {
        "rel_l2_mean": np.mean(all_rel_l2),
        "rel_l2_std": np.std(all_rel_l2),
        "angle_err_mean": np.mean(all_angle_err),
        "angle_err_std": np.std(all_angle_err),
        "latent_codes": latent_codes,
        "deformation_modes": deformation_modes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = InverseOperatorConfig()
    data_config = DatasetConfig(n_samples=args.n_samples, seed=999)

    model = InverseOperatorModel(config).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))

    dataset = DICDataset(data_config)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    evaluate(model, loader, device)


if __name__ == "__main__":
    main()
