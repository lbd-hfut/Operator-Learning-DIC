"""Simple training script — supports Route A, B, C, and D.

Usage:
    python _train_simple.py                         # Route A (default)
    python _train_simple.py --route B               # Route B
    python _train_simple.py --route C               # Route C (U-Net)
    python _train_simple.py --route D               # Route D (ViT Transformer)
"""
import argparse, sys; sys.path.insert(0, ".")
from pathlib import Path
import torch; import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset.folder_dataset import FolderDICDataset
from dataset.collate import collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--route", type=str, default="A", choices=["A", "B", "C", "D"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--data_dir", type=str, default="dataset/dataset/2026-06-01/train")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

device = torch.device("cuda")
is_route_c = args.route == "C"
is_route_d = args.route == "D"

if args.route == "D":
    from dic_vit_method.model import VitDICModel
    from dic_vit_method.config import VitDICConfig
    config = VitDICConfig()
    config.warmup_steps = 0
    model = VitDICModel(config).to(device)
    ckpt_dir = Path("checkpoints/route_d")
elif args.route == "C":
    from dic_unet_method.model import UnetDICModel
    from dic_unet_method.config import UnetDICConfig
    config = UnetDICConfig()
    model = UnetDICModel(config).to(device)
    ckpt_dir = Path("checkpoints/route_c")
elif args.route == "B":
    from deformation_inverse_operator.model import InverseOperatorModel
    from deformation_inverse_operator.config import InverseOperatorConfig
    config = InverseOperatorConfig()
    config.warmup_steps = 0
    config.siamese_downsample = 1
    config.fourier_scale = 2.0
    model = InverseOperatorModel(config).to(device)
    ckpt_dir = Path("checkpoints/route_b")
else:
    from dic_solver_operator.model import SolverOperatorModel
    from dic_solver_operator.config import SolverOperatorConfig
    config = SolverOperatorConfig()
    config.warmup_steps = 0
    config.encoder_downsample = 1
    config.encoder_kernel_size = 3
    config.fourier_scale = 2.0
    model = SolverOperatorModel(config).to(device)
    ckpt_dir = Path("checkpoints/route_a")

dataset = FolderDICDataset(args.data_dir, n_query_min=4096, n_query_max=8192)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

ckpt_dir.mkdir(parents=True, exist_ok=True)
opt = AdamW(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

if is_route_d:
    # Pre-compute bin label helpers for coarse CE loss
    coarse_centers = torch.from_numpy(config.coarse_bin_centers).to(device)


def _bin_label(value, bin_centers):
    """Find index of closest bin center to each value.

    Args:
        value: [N] float tensor
        bin_centers: [n_bins] float tensor

    Returns:
        labels: [N] int64 tensor of bin indices
    """
    # (value.unsqueeze(-1) - centers.unsqueeze(0)).abs().argmin(-1)
    diff = value.unsqueeze(-1) - bin_centers.unsqueeze(0)  # [N, n_bins]
    return diff.abs().argmin(dim=-1)  # [N]


def sample_dense_at_queries(u_dense, qpts):
    """Sample a dense [B, 2, H, W] prediction at query points [B, N_q, 2] → [B, N_q, 2]."""
    B, _, H, W = u_dense.shape
    grid = qpts * 2.0 - 1.0                     # [0,1] → [-1,1]
    grid = grid.unsqueeze(2)                     # [B, N_q, 1, 2]
    u_sampled = F.grid_sample(u_dense, grid, mode="bilinear",
                              padding_mode="border", align_corners=True)
    return u_sampled.squeeze(-1).transpose(1, 2)  # [B, N_q, 2]


model.train()
best_mse = float("inf")
global_step = 0
print(f"Route {args.route} | Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch, {args.epochs} epochs")

for epoch in range(args.epochs):
    epoch_loss_sum = 0.0
    epoch_batches = 0
    step_in_epoch = 0

    for batch in loader:
        ref = batch["ref_img"].to(device)
        tar = batch["tar_img"].to(device)
        qpts = batch["query_points"].to(device)
        u_gt = batch["u_gt"].to(device)
        qmask = batch["query_mask"].to(device)

        opt.zero_grad()

        if is_route_d:
            u_pred, loss_aux = model(ref, tar, qpts)

            # MSE loss on final prediction
            loss_mse = criterion(u_pred[qmask], u_gt[qmask])

            # Coarse CE loss
            u_gt_masked = u_gt[qmask]  # [M, 2]
            u_x_labels = _bin_label(u_gt_masked[:, 0], coarse_centers)
            u_y_labels = _bin_label(u_gt_masked[:, 1], coarse_centers)
            loss_ce_x = F.cross_entropy(
                loss_aux["coarse_logits_x"][qmask], u_x_labels,
            )
            loss_ce_y = F.cross_entropy(
                loss_aux["coarse_logits_y"][qmask], u_y_labels,
            )
            loss_ce = loss_ce_x + loss_ce_y

            # Fine MSE loss
            u_fine_x = loss_aux["u_fine_x"][qmask]
            u_fine_y = loss_aux["u_fine_y"][qmask]
            u_coarse_x = loss_aux["u_coarse_x"][qmask].detach()
            u_coarse_y = loss_aux["u_coarse_y"][qmask].detach()
            loss_fine = criterion(u_fine_x, u_gt_masked[:, 0] - u_coarse_x) \
                        + criterion(u_fine_y, u_gt_masked[:, 1] - u_coarse_y)

            loss = loss_mse + loss_ce + config.fine_loss_weight * loss_fine
        elif is_route_c:
            u_dense = model(ref, tar)            # [B, 2, H, W]
            u_pred = sample_dense_at_queries(u_dense, qpts)
            loss = criterion(u_pred[qmask], u_gt[qmask])
        else:
            u_pred = model(ref, tar, qpts)
            loss = criterion(u_pred[qmask], u_gt[qmask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        epoch_loss_sum += loss.item()
        epoch_batches += 1

        if step_in_epoch % 200 == 0:
            u_pred_masked = u_pred[qmask]
            u_gt_masked = u_gt[qmask]
            mae = (u_pred_masked - u_gt_masked).abs().mean().item()
            pred_std = u_pred_masked.std().item()
            gt_std = u_gt_masked.std().item()
            zero_mse = (u_gt_masked ** 2).mean().item()
            mse_val = loss.item()
            print(f"epoch [{epoch:03d}][{step_in_epoch:04d}] "
                  f"MSE={mse_val:.6f} (zero={zero_mse:.4f}) "
                  f"MAE={mae:.4f} | pred=[{u_pred.min():.2f},{u_pred.max():.2f}] "
                  f"σ={pred_std:.3f} gt_σ={gt_std:.3f}")
            if mse_val < best_mse:
                best_mse = mse_val
                torch.save({"model_state_dict": model.state_dict(), "config": model.config},
                           str(ckpt_dir / "best.pt"))
                print(f"  -> saved best (MSE={best_mse:.6f})")

        global_step += 1
        step_in_epoch += 1

    # --- End of epoch ---
    if epoch_batches > 0:
        avg_loss = epoch_loss_sum / epoch_batches
        print(f"epoch [{epoch:03d}] avg MSE={avg_loss:.6f}")

torch.save({"model_state_dict": model.state_dict(), "config": model.config},
           str(ckpt_dir / "last.pt"))
print(f"Done. Final MSE={loss.item():.6f}, best MSE={best_mse:.6f}")
print(f"Checkpoints: {ckpt_dir / 'best.pt'}, {ckpt_dir / 'last.pt'}")
