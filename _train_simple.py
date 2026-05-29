"""Simple training script — supports Route A and Route B.

Usage:
    python _train_simple.py                         # Route A (default)
    python _train_simple.py --route B               # Route B
"""
import argparse, sys; sys.path.insert(0, ".")
from pathlib import Path
import torch; import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset.folder_dataset import FolderDICDataset
from dataset.collate import collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--route", type=str, default="A", choices=["A", "B"])
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--data_dir", type=str, default="dataset/dataset/2026-05-27/train")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

device = torch.device("cuda")

if args.route == "B":
    from deformation_inverse_operator.model import InverseOperatorModel
    from deformation_inverse_operator.config import InverseOperatorConfig
    config = InverseOperatorConfig()
    config.warmup_steps = 0
    config.siamese_downsample = 1        # H/2 = 128x128 — preserve speckle detail
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

print(f"Route {args.route} | Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Dataset: {len(dataset)} samples, {len(loader)} batches")

model.train()
best_mse = float("inf")
step = 0
while step < args.steps:
    for batch in loader:
        if step >= args.steps:
            break
        ref = batch["ref_img"].to(device)
        tar = batch["tar_img"].to(device)
        qpts = batch["query_points"].to(device)
        u_gt = batch["u_gt"].to(device)
        qmask = batch["query_mask"].to(device)

        opt.zero_grad()
        u_pred = model(ref, tar, qpts)
        loss = criterion(u_pred[qmask], u_gt[qmask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if step % 200 == 0:
            u_pred_masked = u_pred[qmask]
            u_gt_masked = u_gt[qmask]
            mae = (u_pred_masked - u_gt_masked).abs().mean().item()
            pred_std = u_pred_masked.std().item()
            gt_std = u_gt_masked.std().item()
            zero_mse = (u_gt_masked ** 2).mean().item()
            mse_val = loss.item()
            print(f"step {step:5d}: MSE={mse_val:.6f} (zero={zero_mse:.4f}) "
                  f"MAE={mae:.4f} | pred=[{u_pred.min():.2f},{u_pred.max():.2f}] "
                  f"σ={pred_std:.3f} gt_σ={gt_std:.3f}")
            if mse_val < best_mse:
                best_mse = mse_val
                torch.save({"model_state_dict": model.state_dict(), "config": model.config},
                           str(ckpt_dir / "best.pt"))
                print(f"  -> saved best (MSE={best_mse:.6f})")
        step += 1

torch.save({"model_state_dict": model.state_dict(), "config": model.config},
           str(ckpt_dir / "last.pt"))
print(f"Done. Final MSE={loss.item():.6f}, best MSE={best_mse:.6f}")
print(f"Checkpoints: {ckpt_dir / 'best.pt'}, {ckpt_dir / 'last.pt'}")
