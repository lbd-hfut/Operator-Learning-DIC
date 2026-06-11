"""Test irregular-ROI prediction: zero out ref outside ROI, warp to create tar, predict.

Route A, B, C are evaluated side-by-side.
"""
import sys; sys.path.insert(0, ".")
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── helpers ──────────────────────────────────────────────────────────

def load_sample(data_dir, idx):
    data_dir = Path(data_dir)
    ref = np.array(Image.open(data_dir / "ref" / f"{idx:06d}.png"), dtype=np.float32) / 255.0
    tar = np.array(Image.open(data_dir / "tar" / f"{idx:06d}.png"), dtype=np.float32) / 255.0
    u_field = np.load(data_dir / "u_field" / f"{idx:06d}.npy")
    roi_path = data_dir / "roi_mask" / f"{idx:06d}.png"
    roi = np.array(Image.open(roi_path)) > 127 if roi_path.exists() else np.ones(ref.shape, dtype=bool)
    return ref, tar, u_field, roi


def warp_image(ref, u_field):
    H, W = ref.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    src_y = yy - u_field[..., 1]
    src_x = xx - u_field[..., 0]
    coords = np.stack([src_y.ravel(), src_x.ravel()], axis=0)
    return map_coordinates(ref, coords, order=1, mode="constant", cval=0.0).reshape(H, W)


def run_prediction(route, ckpt_path, ref, tar, roi, device):
    import torch
    if route == "D":
        from dic_vit_method.predict import Predictor
    elif route == "C":
        from dic_unet_method.predict import Predictor
    elif route == "B":
        from deformation_inverse_operator.predict import Predictor
    else:
        from dic_solver_operator.predict import Predictor

    pred = Predictor(str(ckpt_path), device)
    u_pred = pred.dense(ref, tar)

    m = roi
    mae = np.abs(u_pred[m] - u_field_gt[m]).mean()
    mse = ((u_pred[m] - u_field_gt[m]) ** 2).mean()
    zero_mse = (u_field_gt[m] ** 2).mean()
    return u_pred, mae, mse, zero_mse


def make_synthetic_roi(shape, roi_type="circle", **kwargs):
    H, W = shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = H / 2, W / 2

    if roi_type == "circle":
        r = kwargs.get("radius", min(H, W) * 0.35)
        roi = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
    elif roi_type == "ellipse":
        rx = kwargs.get("rx", W * 0.35)
        ry = kwargs.get("ry", H * 0.25)
        roi = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) < 1
    elif roi_type == "ring":
        r_outer = kwargs.get("r_outer", min(H, W) * 0.40)
        r_inner = kwargs.get("r_inner", min(H, W) * 0.15)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        roi = (d2 < r_outer ** 2) & (d2 > r_inner ** 2)
    elif roi_type == "notch":
        margin = int(H * 0.05)
        notch_w = int(W * 0.35)
        roi = np.zeros(shape, dtype=bool)
        roi[margin:-margin, :] = True
        roi[:, :notch_w] = False
        roi[:, -notch_w:] = False
        hole_r = kwargs.get("hole_r", int(min(H, W) * 0.08))
        hole = (yy - cy) ** 2 + (xx - cx) ** 2 < hole_r ** 2
        roi = roi & ~hole
    else:
        raise ValueError(f"Unknown roi_type: {roi_type}")
    return roi


# ── plot ─────────────────────────────────────────────────────────────

def plot_all(ref_masked, tar_warped, tar_orig, roi, u_gt,
             results,  # dict: route_label -> (u_pred, mae, mse, zero, ratio)
             idx, save_path):
    """3×6 figure: images row + u_x row + u_y row.  Cols: GT, A, B, C, D, Error."""
    route_order = ["A", "B", "C", "D"]
    n_routes = len(route_order)
    n_cols = n_routes + 2  # GT + N routes + Error
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14))
    fig.suptitle(f"Irregular-ROI Prediction — Sample {idx:06d}", fontsize=14, fontweight="bold")

    mask = ~roi

    # Row 0: images
    axes[0, 0].imshow(ref_masked, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Ref (ROI-masked)")
    axes[0, 1].imshow(tar_warped, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Tar (warped)")
    axes[0, 2].imshow(tar_orig, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Tar (original)")
    diff_full = np.abs(tar_warped - ref_masked)
    axes[0, 3].imshow(diff_full, cmap="hot")
    axes[0, 3].set_title("|Tar - Ref|")
    axes[0, n_cols - 1].imshow(roi, cmap="gray")
    axes[0, n_cols - 1].set_title(f"ROI ({roi.sum()} px)")

    # Determine colorbar range
    vx = max(abs(u_gt[..., 0][roi]).max(), 0.5)
    vy = max(abs(u_gt[..., 1][roi]).max(), 0.5)
    for _, (u, _, _, _, _) in results.items():
        vx = max(vx, abs(u[..., 0][roi]).max())
        vy = max(vy, abs(u[..., 1][roi]).max())

    # Row 1: u_x  (cols: GT, A, B, C, D, Error)
    axes[1, 0].imshow(np.ma.array(u_gt[..., 0], mask=mask), cmap="RdBu_r", vmin=-vx, vmax=vx)
    axes[1, 0].set_title(f"GT u_x\n[{u_gt[..., 0].min():.2f},{u_gt[..., 0].max():.2f}]")
    for j, route in enumerate(route_order):
        u, mae, _, _, _ = results.get(route, (None, None, None, None, None))
        if u is not None:
            ax = axes[1, j + 1]
            ax.imshow(np.ma.array(u[..., 0], mask=mask), cmap="RdBu_r", vmin=-vx, vmax=vx)
            ax.set_title(f"Route {route} u_x\nMAE={mae:.3f}")
    # Last col: u_x error of Route A
    u_a = results.get("A", (None,))[0]
    if u_a is not None:
        err = np.ma.array(u_a[..., 0] - u_gt[..., 0], mask=mask)
        ve = max(abs(err).max(), 0.1)
        im = axes[1, n_cols - 1].imshow(err, cmap="RdBu_r", vmin=-ve, vmax=ve)
        axes[1, n_cols - 1].set_title(f"Error u_x (A)")
        plt.colorbar(im, ax=axes[1, n_cols - 1], shrink=0.8)

    # Row 2: u_y
    axes[2, 0].imshow(np.ma.array(u_gt[..., 1], mask=mask), cmap="RdBu_r", vmin=-vy, vmax=vy)
    axes[2, 0].set_title(f"GT u_y\n[{u_gt[..., 1].min():.2f},{u_gt[..., 1].max():.2f}]")
    for j, route in enumerate(route_order):
        u, mae, _, _, _ = results.get(route, (None, None, None, None, None))
        if u is not None:
            ax = axes[2, j + 1]
            ax.imshow(np.ma.array(u[..., 1], mask=mask), cmap="RdBu_r", vmin=-vy, vmax=vy)
            ax.set_title(f"Route {route} u_y\nMAE={mae:.3f}")
    # Last col: u_y error of Route A
    if u_a is not None:
        err = np.ma.array(u_a[..., 1] - u_gt[..., 1], mask=mask)
        ve = max(abs(err).max(), 0.1)
        im = axes[2, n_cols - 1].imshow(err, cmap="RdBu_r", vmin=-ve, vmax=ve)
        axes[2, n_cols - 1].set_title(f"Error u_y (A)")
        plt.colorbar(im, ax=axes[2, n_cols - 1], shrink=0.8)

    # Summary text
    lines = []
    for route in route_order:
        r = results.get(route)
        if r is not None:
            _, mae, mse, zero, ratio = r
            lines.append(f"Route {route}: MAE={mae:.4f}  MSE={mse:.6f}  zero={zero:.6f}  ratio={ratio:.4f}")
    fig.text(0.5, 0.01, "  |  ".join(lines), ha="center", fontsize=9, family="monospace")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Irregular-ROI prediction test (Route A/B/C/D)")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="dataset/dataset/2026-05-27/test")
    parser.add_argument("--ckpt_a", type=str, default="checkpoints/route_a/best.pt")
    parser.add_argument("--ckpt_b", type=str, default="checkpoints/route_b/best.pt")
    parser.add_argument("--ckpt_c", type=str, default="checkpoints/route_c/best.pt")
    parser.add_argument("--ckpt_d", type=str, default="checkpoints/route_d/best.pt")
    parser.add_argument("--save_plot", type=str, default="predictions/irregular_roi.png")
    parser.add_argument("--roi_type", type=str, default=None,
                        choices=["circle", "ellipse", "ring", "notch"])
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load
    print(f"Loading sample {args.sample:06d} ...")
    ref_orig, tar_orig, u_field_gt_orig, roi = load_sample(args.data_dir, args.sample)

    # 2. Override ROI
    if args.roi_type:
        roi = make_synthetic_roi(ref_orig.shape, roi_type=args.roi_type)
        print(f"Using synthetic ROI '{args.roi_type}': {roi.sum()} pixels in ROI")

    # 3. Mask & warp
    ref_masked = ref_orig.copy()
    ref_masked[~roi] = 0.0
    print("Warping masked reference to create synthetic target ...")
    tar_warped = warp_image(ref_masked, u_field_gt_orig)

    global u_field_gt
    u_field_gt = u_field_gt_orig

    print(f"  ref_masked:  [{ref_masked.min():.3f}, {ref_masked.max():.3f}]")
    print(f"  tar_warped:  [{tar_warped.min():.3f}, {tar_warped.max():.3f}]")
    print(f"  u_field GT:  x=[{u_field_gt[...,0].min():.3f}, {u_field_gt[...,0].max():.3f}] "
          f"y=[{u_field_gt[...,1].min():.3f}, {u_field_gt[...,1].max():.3f}]")
    print(f"  ROI pixels:  {roi.sum()}/{roi.size}")

    # 4. Run all three routes
    results = {}
    for route, ckpt_path in [("A", args.ckpt_a), ("B", args.ckpt_b), ("C", args.ckpt_c), ("D", args.ckpt_d)]:
        if not Path(ckpt_path).exists():
            print(f"\nRoute {route}: checkpoint not found ({ckpt_path}), skipping")
            continue
        print(f"\nRoute {route} prediction ...")
        u, mae, mse, zero = run_prediction(route, ckpt_path, ref_masked, tar_warped, roi, device)
        ratio = mse / zero if zero > 0 else float("inf")
        print(f"  MAE={mae:.4f}  MSE={mse:.6f}  zero={zero:.6f}  ratio={ratio:.4f}")
        results[route] = (u, mae, mse, zero, ratio)

    # 5. Plot
    print("\nPlotting ...")
    plot_all(ref_masked, tar_warped, tar_orig, roi, u_field_gt,
             results, args.sample, args.save_plot)


if __name__ == "__main__":
    main()
