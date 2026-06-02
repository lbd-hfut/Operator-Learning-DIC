"""Multi-resolution prediction test for Route A/B.

Tests models trained at 256×256 on images at various resolutions.
Uses synthetic analytic displacement fields for exact ground truth at any resolution.

Usage:
    python _test_multires.py                          # default resolutions
    python _test_multires.py --resolutions 128,256,512  # custom
    python _test_multires.py --sample 5 --amplitude 5.0
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


# ── Synthetic displacement field ────────────────────────────────────────

def make_synthetic_displacement(H, W, amplitude=3.0):
    """Generate a smooth, multi-frequency analytic displacement field.

    Returns [H, W, 2] in pixel units. Exactly known — no warp error.
    """
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")

    # Normalized coordinates
    yn, xn = yy / max(H - 1, 1), xx / max(W - 1, 1)

    u_x = amplitude * (
        np.sin(2 * np.pi * xn * 1.5) * np.cos(2 * np.pi * yn * 1.0) +
        0.7 * np.sin(2 * np.pi * xn * 3.0 + 0.5) * np.cos(2 * np.pi * yn * 2.5) +
        0.4 * np.cos(2 * np.pi * xn * 0.7) * np.sin(2 * np.pi * yn * 3.2)
    )
    u_y = amplitude * (
        np.cos(2 * np.pi * xn * 1.2) * np.sin(2 * np.pi * yn * 1.8) +
        0.6 * np.sin(2 * np.pi * xn * 2.7) * np.cos(2 * np.pi * yn * 3.0) +
        0.5 * np.sin(2 * np.pi * xn * 0.8 + 1.0) * np.cos(2 * np.pi * yn * 1.3)
    )
    return np.stack([u_x, u_y], axis=-1)


# ── Image warp ──────────────────────────────────────────────────────────

def warp_image(img, u_field):
    """Warp image by displacement field (inverse warp: src = dst - u)."""
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    src_y = yy - u_field[..., 1]
    src_x = xx - u_field[..., 0]
    coords = np.stack([src_y.ravel(), src_x.ravel()], axis=0)
    return map_coordinates(img, coords, order=1, mode="constant", cval=0.0).reshape(H, W)


# ── Prediction ──────────────────────────────────────────────────────────

def run_prediction(route, ckpt_path, ref, tar, device):
    import torch
    if route == "B":
        from deformation_inverse_operator.predict import Predictor
    else:
        from dic_solver_operator.predict import Predictor

    pred = Predictor(str(ckpt_path), device)
    u_pred = pred.dense(ref, tar)

    H, W = ref.shape
    u_gt = make_synthetic_displacement(H, W)

    mae = np.abs(u_pred - u_gt).mean()
    mse = ((u_pred - u_gt) ** 2).mean()
    zero_mse = (u_gt ** 2).mean()

    return u_pred, u_gt, mae, mse, zero_mse


# ── Plotting ────────────────────────────────────────────────────────────

def plot_results(results, sample_idx, ref_img, save_path):
    """2×N grid: ref/tar row + u_x error row for each resolution."""
    res_list = list(results.keys())
    n = len(res_list)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Multi-Resolution Test — Sample {sample_idx:06d}", fontsize=13, fontweight="bold")

    route_order = ["A", "B"]
    cmaps = {"A": "RdBu_r", "B": "RdBu_r"}

    for j, res_name in enumerate(res_list):
        data = results[res_name]
        H, W = data["H"], data["W"]

        # Row 0: reference image
        axes[0, j].imshow(data["ref"], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"{res_name} ({H}×{W})\nRef (resized)")

        # Row 1: displacement magnitude error (mean across routes)
        # Actually show Route A u_x prediction vs GT
        if "A" in data["routes"]:
            u_gt = data["routes"]["A"]["u_gt"]
            u_pred = data["routes"]["A"]["u_pred"]
            err_x = u_pred[..., 0] - u_gt[..., 0]
            vmax = max(abs(err_x).max(), 0.5)
            im = axes[1, j].imshow(err_x, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[1, j].set_title(f"Route A u_x Error\nMAE={data['routes']['A']['mae']:.3f} px")
            plt.colorbar(im, ax=axes[1, j], shrink=0.85)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Summary line
    lines = []
    for res_name in res_list:
        parts = [f"{res_name}:"]
        for route in route_order:
            r = results[res_name]["routes"].get(route)
            if r:
                parts.append(f"R{route} MAE={r['mae']:.3f} MSE={r['mse']:.4f}")
        lines.append("  ".join(parts))
    fig.text(0.5, 0.01, " | ".join(lines), ha="center", fontsize=7, family="monospace")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_mae_curve(results, save_path):
    """Plot MAE vs resolution as a line chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("MAE vs Resolution", fontsize=13, fontweight="bold")

    res_labels = list(results.keys())
    x = np.arange(len(res_labels))

    for route, color, marker in [("A", "#2196F3", "o"), ("B", "#FF5722", "s")]:
        maes = []
        for res_name in res_labels:
            r = results[res_name]["routes"].get(route)
            maes.append(r["mae"] if r else None)
        valid_idx = [i for i, m in enumerate(maes) if m is not None]
        if valid_idx:
            xi = [x[i] for i in valid_idx]
            mi = [maes[i] for i in valid_idx]
            ax.plot(xi, mi, color=color, marker=marker, linewidth=2,
                    markersize=8, label=f"Route {route}")

    ax.set_xticks(x)
    ax.set_xticklabels(res_labels, rotation=30, ha="right")
    ax.set_ylabel("MAE (pixels)")
    ax.set_xlabel("Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add pixel-count annotations
    for i, res_name in enumerate(res_labels):
        H, W = results[res_name]["H"], results[res_name]["W"]
        ax.annotate(f"{H}×{W}", (i, ax.get_ylim()[1] * 0.95),
                    ha="center", fontsize=7, color="gray")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────

def parse_resolutions(arg):
    return [tuple(map(int, r.split("x"))) for r in arg.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Multi-resolution prediction test")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="dataset/dataset/2026-05-27/test")
    parser.add_argument("--ckpt_a", type=str, default="checkpoints/route_a/best.pt")
    parser.add_argument("--ckpt_b", type=str, default="checkpoints/route_b/best.pt")
    parser.add_argument("--amplitude", type=float, default=3.0,
                        help="Max displacement amplitude in pixels")
    parser.add_argument("--resolutions", type=str, default="128x128,256x256,384x384,384x768,512x512,640x640",
                        help="Comma-separated HxW pairs")
    parser.add_argument("--save_plot", type=str, default="predictions/multires_grid.png")
    parser.add_argument("--save_curve", type=str, default="predictions/multires_curve.png")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse resolutions
    res_list = parse_resolutions(args.resolutions)
    res_names = [f"{h}x{w}" for h, w in res_list]

    # Load reference image
    print(f"Loading sample {args.sample:06d} ...")
    data_dir = Path(args.data_dir)
    ref_path = data_dir / "ref" / f"{args.sample:06d}.png"
    ref_img = np.array(Image.open(ref_path), dtype=np.float32) / 255.0
    print(f"  Reference: {ref_img.shape}")

    results = {}

    for (H, W), res_name in zip(res_list, res_names):
        print(f"\n{'='*60}")
        print(f"Testing resolution: {H}×{W} ({res_name})")
        print(f"{'='*60}")

        # Resize reference to target resolution
        from PIL import Image as PILImage
        ref_resized = np.array(
            PILImage.fromarray((ref_img * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0

        # Generate synthetic displacement and warp
        u_syn = make_synthetic_displacement(H, W, args.amplitude)
        tar_warped = warp_image(ref_resized, u_syn)

        print(f"  u_syn: x=[{u_syn[..., 0].min():.2f}, {u_syn[..., 0].max():.2f}] "
              f"y=[{u_syn[..., 1].min():.2f}, {u_syn[..., 1].max():.2f}]")

        entry = {"H": H, "W": W, "ref": ref_resized, "tar": tar_warped, "routes": {}}

        for route, ckpt_path in [("A", args.ckpt_a), ("B", args.ckpt_b)]:
            if not Path(ckpt_path).exists():
                print(f"  Route {route}: checkpoint not found ({ckpt_path}), skipping")
                continue
            print(f"  Route {route} predicting ...")
            u_pred, u_gt, mae, mse, zero = run_prediction(route, ckpt_path,
                                                          ref_resized, tar_warped, device)
            ratio = mse / zero if zero > 0 else float("inf")
            print(f"    MAE={mae:.4f}  MSE={mse:.6f}  zero={zero:.6f}  ratio={ratio:.4f}")
            entry["routes"][route] = {
                "u_pred": u_pred, "u_gt": u_gt,
                "mae": mae, "mse": mse, "zero": zero, "ratio": ratio,
            }

        results[res_name] = entry

    # Report summary table
    print(f"\n{'='*60}")
    print("Summary: MAE (pixels) vs Resolution")
    print(f"{'='*60}")
    header = f"{'Resolution':>12} | {'Route A':>10} {'Route B':>10} | {'Ratio_A':>8} {'Ratio_B':>8}"
    print(header)
    print("-" * len(header))
    for res_name in res_names:
        entry = results[res_name]
        ma = entry["routes"].get("A", {}).get("mae", None)
        mb = entry["routes"].get("B", {}).get("mae", None)
        ra = entry["routes"].get("A", {}).get("ratio", None)
        rb = entry["routes"].get("B", {}).get("ratio", None)
        ma_s = f"{ma:.4f}" if ma is not None else "N/A"
        mb_s = f"{mb:.4f}" if mb is not None else "N/A"
        ra_s = f"{ra:.4f}" if ra is not None else "N/A"
        rb_s = f"{rb:.4f}" if rb is not None else "N/A"
        print(f"{res_name:>12} | {ma_s:>10} {mb_s:>10} | {ra_s:>8} {rb_s:>8}")

    # Plots
    print("\nPlotting ...")
    plot_results(results, args.sample, ref_img, args.save_plot)
    plot_mae_curve(results, args.save_curve)


if __name__ == "__main__":
    main()
