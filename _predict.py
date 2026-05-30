"""Predict displacement field and visualize vs ground truth.

Usage:
    python _predict.py                                    # Route A, auto-train if no ckpt
    python _predict.py --route B                          # Route B
    python _predict.py --route C                          # Route C (U-Net)
    python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample 0
"""
import argparse
import sys; sys.path.insert(0, ".")
import numpy as np
import torch; import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_model(route, device):
    if route == "C":
        from dic_unet_method.model import UnetDICModel
        from dic_unet_method.config import UnetDICConfig
        config = UnetDICConfig()
        return UnetDICModel(config).to(device)
    elif route == "B":
        from deformation_inverse_operator.model import InverseOperatorModel
        from deformation_inverse_operator.config import InverseOperatorConfig
        config = InverseOperatorConfig()
        config.siamese_downsample = 1
        config.fourier_scale = 2.0
        return InverseOperatorModel(config).to(device)
    else:
        from dic_solver_operator.model import SolverOperatorModel
        from dic_solver_operator.config import SolverOperatorConfig
        config = SolverOperatorConfig()
        config.encoder_downsample = 1
        config.encoder_kernel_size = 3
        config.fourier_scale = 2.0
        return SolverOperatorModel(config).to(device)


def load_sample(data_dir, idx):
    data_dir = Path(data_dir)
    ref = np.array(Image.open(data_dir / "ref" / f"{idx:06d}.png"), dtype=np.float32) / 255.0
    tar = np.array(Image.open(data_dir / "tar" / f"{idx:06d}.png"), dtype=np.float32) / 255.0
    u_field = np.load(data_dir / "u_field" / f"{idx:06d}.npy")
    roi_path = data_dir / "roi_mask" / f"{idx:06d}.png"
    roi = np.array(Image.open(roi_path)) > 127 if roi_path.exists() else np.ones(u_field.shape[:2], dtype=bool)
    return ref, tar, u_field, roi


def build_dense_queries(H, W, device):
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)


@torch.no_grad()
def predict_dense(model, ref, tar, device, route="A"):
    model.eval()
    ref_t = torch.from_numpy(ref).unsqueeze(0).unsqueeze(0).to(device)
    tar_t = torch.from_numpy(tar).unsqueeze(0).unsqueeze(0).to(device)
    H, W = ref.shape

    if route == "C":
        # UNet: direct dense output [1, 2, H, W]
        u_pred = model(ref_t, tar_t)
        return u_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

    queries = build_dense_queries(H, W, device)
    encoded = model.encode(ref_t, tar_t)
    if route == "B":
        u_pred = model.decode(queries, *encoded)
    else:
        u_pred = model.decode(queries, encoded)
    return u_pred.reshape(H, W, 2).cpu().numpy()


def plot_results(ref, tar, u_pred, u_gt, roi, idx, route, save_path=None):
    H, W = ref.shape
    u_pred_x, u_pred_y = u_pred[..., 0], u_pred[..., 1]
    u_gt_x, u_gt_y = u_gt[..., 0], u_gt[..., 1]
    mask = ~roi
    upx = np.ma.array(u_pred_x, mask=mask); upy = np.ma.array(u_pred_y, mask=mask)
    ugx = np.ma.array(u_gt_x, mask=mask); ugy = np.ma.array(u_gt_y, mask=mask)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"Route {route} — Sample {idx:06d}", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(ref, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Reference Image")
    axes[0, 1].imshow(tar, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Target (Deformed) Image")
    axes[0, 2].imshow(np.abs(tar - ref), cmap="hot")
    axes[0, 2].set_title("|Tar - Ref| Difference")

    vx = max(abs(ugx).max(), abs(upx).max(), 0.5)
    for a, d, t in [(axes[1, 0], upx, "Pred u_x"), (axes[1, 1], ugx, "GT u_x"), (axes[1, 2], upx - ugx, "Error u_x")]:
        im = a.imshow(d, cmap="RdBu_r", vmin=-vx, vmax=vx)
        a.set_title(f"{t}  [{d.min():.2f}, {d.max():.2f}]")
        plt.colorbar(im, ax=a, shrink=0.8)

    vy = max(abs(ugy).max(), abs(upy).max(), 0.5)
    for a, d, t in [(axes[2, 0], upy, "Pred u_y"), (axes[2, 1], ugy, "GT u_y"), (axes[2, 2], upy - ugy, "Error u_y")]:
        im = a.imshow(d, cmap="RdBu_r", vmin=-vy, vmax=vy)
        a.set_title(f"{t}  [{d.min():.2f}, {d.max():.2f}]")
        plt.colorbar(im, ax=a, shrink=0.8)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        out = Path(save_path)
    else:
        out = Path("predictions") / f"route_{route.lower()}_{idx:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def train_and_save(data_dir, ckpt_path, route, steps=3000):
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from dataset.folder_dataset import FolderDICDataset
    from dataset.collate import collate_fn

    device = torch.device("cuda")
    dataset = FolderDICDataset(str(data_dir), n_query_min=4096, n_query_max=8192)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    model = make_model(route, device)
    opt = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    is_route_c = route == "C"

    print(f"Training Route {route}, {steps} steps...")
    model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            ref = batch["ref_img"].to(device)
            tar = batch["tar_img"].to(device)
            qpts = batch["query_points"].to(device)
            u_gt = batch["u_gt"].to(device)
            qmask = batch["query_mask"].to(device)
            opt.zero_grad()

            if is_route_c:
                u_dense = model(ref, tar)
                grid = qpts * 2.0 - 1.0
                grid = grid.unsqueeze(2)
                u_sampled = F.grid_sample(u_dense, grid, mode="bilinear",
                                          padding_mode="border", align_corners=True)
                u_pred = u_sampled.squeeze(-1).transpose(1, 2)
            else:
                u_pred = model(ref, tar, qpts)

            loss = criterion(u_pred[qmask], u_gt[qmask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            if step % 500 == 0:
                mae = (u_pred[qmask] - u_gt[qmask]).abs().mean().item()
                print(f"  step {step}: MSE={loss.item():.6f} MAE={mae:.4f}")
            step += 1

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, str(ckpt_path))
    print(f"Saved: {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="dataset/dataset/2026-05-27/test")
    parser.add_argument("--save_plot", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt and Path(args.ckpt).exists():
        print(f"Loading checkpoint: {args.ckpt}")
        model = make_model(args.route, device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        ckpt_path = args.ckpt or f"checkpoints/route_{args.route.lower()}/quick_model.pt"
        model = train_and_save(args.data_dir.replace("/test", "/train"), ckpt_path, args.route)

    print(f"\nLoading sample {args.sample:06d} from {args.data_dir}...")
    ref, tar, u_field, roi = load_sample(args.data_dir, args.sample)

    print("Predicting dense displacement...")
    u_pred = predict_dense(model, ref, tar, device, route=args.route)

    m = roi
    mae = np.abs(u_pred[m] - u_field[m]).mean()
    mse = ((u_pred[m] - u_field[m]) ** 2).mean()
    zero_mse = (u_field[m] ** 2).mean()
    print(f"  Dense MAE: {mae:.4f} px  |  MSE: {mse:.6f} (zero baseline: {zero_mse:.6f})")
    print(f"  Pred u_x: [{u_pred[...,0].min():.3f}, {u_pred[...,0].max():.3f}]")
    print(f"  GT   u_x: [{u_field[...,0].min():.3f}, {u_field[...,0].max():.3f}]")

    print("Plotting...")
    plot_results(ref, tar, u_pred, u_field, roi, args.sample, args.route, save_path=args.save_plot)


if __name__ == "__main__":
    main()
