"""Multi-route prediction and evaluation for FEM test cases.

Loads generated test images, runs Route A/B/C predictors,
computes MAE/MSE within ROI, and generates comparison figures.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_predictions(ref, tar, roi, ckpt_a=None, ckpt_b=None, ckpt_c=None, device="cuda"):
    """Run all three route predictions on a test case.

    Args:
        ref: [H, W] float32 reference image (ROI-masked)
        tar: [H, W] float32 target image
        roi: [H, W] bool valid ROI mask
        ckpt_a/b/c: checkpoint paths (uses defaults if None)
        device: 'cuda' or 'cpu'

    Returns:
        results: dict mapping route label -> (u_pred, mae, mse)
    """
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("  CUDA not available, falling back to CPU")

    ckpt_a = ckpt_a or _project_root / "checkpoints" / "route_a" / "best.pt"
    ckpt_b = ckpt_b or _project_root / "checkpoints" / "route_b" / "best.pt"
    ckpt_c = ckpt_c or _project_root / "checkpoints" / "route_c" / "best.pt"

    results = {}

    # Route A
    ckpt_a = Path(ckpt_a)
    if ckpt_a.exists():
        from dic_solver_operator.predict import Predictor as PredA
        print(f"  Route A: loading {ckpt_a} ...")
        pred_a = PredA(str(ckpt_a), device)
        u_a = pred_a.dense(ref, tar)
        mae_a = np.abs(u_a[roi] - _u_gt[roi]).mean() if hasattr(_u_gt, 'shape') else 0
        mse_a = ((u_a[roi] - _u_gt[roi]) ** 2).mean() if hasattr(_u_gt, 'shape') else 0
        results["A"] = (u_a, mae_a, mse_a)
        print(f"    MAE={mae_a:.4f}  MSE={mse_a:.6f}")
    else:
        print(f"  Route A: checkpoint not found ({ckpt_a}), skipping")

    # Route B
    ckpt_b = Path(ckpt_b)
    if ckpt_b.exists():
        from deformation_inverse_operator.predict import Predictor as PredB
        print(f"  Route B: loading {ckpt_b} ...")
        pred_b = PredB(str(ckpt_b), device)
        u_b = pred_b.dense(ref, tar)
        mae_b = np.abs(u_b[roi] - _u_gt[roi]).mean() if hasattr(_u_gt, 'shape') else 0
        mse_b = ((u_b[roi] - _u_gt[roi]) ** 2).mean() if hasattr(_u_gt, 'shape') else 0
        results["B"] = (u_b, mae_b, mse_b)
        print(f"    MAE={mae_b:.4f}  MSE={mse_b:.6f}")
    else:
        print(f"  Route B: checkpoint not found ({ckpt_b}), skipping")

    # Route C
    ckpt_c = Path(ckpt_c)
    if ckpt_c.exists():
        from dic_unet_method.predict import Predictor as PredC
        print(f"  Route C: loading {ckpt_c} ...")
        pred_c = PredC(str(ckpt_c), device)
        u_c = pred_c.dense(ref, tar)
        mae_c = np.abs(u_c[roi] - _u_gt[roi]).mean() if hasattr(_u_gt, 'shape') else 0
        mse_c = ((u_c[roi] - _u_gt[roi]) ** 2).mean() if hasattr(_u_gt, 'shape') else 0
        results["C"] = (u_c, mae_c, mse_c)
        print(f"    MAE={mae_c:.4f}  MSE={mse_c:.6f}")
    else:
        print(f"  Route C: checkpoint not found ({ckpt_c}), skipping")

    return results


# Module-level global for storing ground truth during evaluation
_u_gt = None


def evaluate_case(case_name, data_dict, ckpt_a=None, ckpt_b=None, ckpt_c=None,
                  device="cuda", save_plot=None):
    """Evaluate a single test case across all routes.

    Args:
        case_name: 'circle', 'ring', 'notch' (for plot title)
        data_dict: dict from generate_test_case with ref_array, tar_array,
                   u_field_array, roi_array
        ckpt_a/b/c: checkpoint paths
        device: 'cuda' or 'cpu'
        save_plot: path to save figure (or None)

    Returns:
        results dict
    """
    global _u_gt
    ref = data_dict["ref_array"]
    tar = data_dict["tar_array"]
    u_gt = data_dict["u_field_array"]
    roi = data_dict["roi_array"]
    _u_gt = u_gt

    print(f"\n{'='*60}")
    print(f"Evaluating: {case_name}")
    print(f"  ROI pixels: {roi.sum()}/{roi.size}  "
          f"u_x: [{u_gt[..., 0][roi].min():.2f}, {u_gt[..., 0][roi].max():.2f}]  "
          f"u_y: [{u_gt[..., 1][roi].min():.2f}, {u_gt[..., 1][roi].max():.2f}]")

    results = run_predictions(ref, tar, roi, ckpt_a, ckpt_b, ckpt_c, device)

    if save_plot:
        _plot_results(case_name, ref, tar, u_gt, roi, results, save_plot)

    _u_gt = None
    return results


def _plot_results(case_name, ref, tar, u_gt, roi, results, save_path):
    """Generate 3-row comparison figure."""
    route_order = ["A", "B", "C"]
    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    fig.suptitle(f"FEM-based Irregular-ROI: {case_name.title()}", fontsize=14, fontweight="bold")

    mask = ~roi

    # Row 0: Images
    axes[0, 0].imshow(ref, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Ref (ROI-masked)")
    axes[0, 1].imshow(tar, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Tar (warped)")
    diff_img = np.abs(tar - ref)
    axes[0, 2].imshow(diff_img, cmap="hot")
    axes[0, 2].set_title(f"|Tar - Ref|")
    axes[0, 3].imshow(roi, cmap="gray")
    axes[0, 3].set_title(f"ROI ({roi.sum()} px)")
    # FEM u_x overview in last col of row 0
    vx = max(abs(u_gt[..., 0][roi]).max(), 0.5)
    axes[0, 4].imshow(np.ma.array(u_gt[..., 0], mask=mask), cmap="RdBu_r", vmin=-vx, vmax=vx)
    axes[0, 4].set_title(f"FEM GT u_x\n[{u_gt[..., 0][roi].min():.2f},{u_gt[..., 0][roi].max():.2f}]")

    # Determine colorbar ranges from all results
    vx = max(abs(u_gt[..., 0][roi]).max(), 0.5)
    vy = max(abs(u_gt[..., 1][roi]).max(), 0.5)
    for _, (u, _, _) in results.items():
        vx = max(vx, abs(u[..., 0][roi]).max())
        vy = max(vy, abs(u[..., 1][roi]).max())

    # Row 1: u_x
    axes[1, 0].imshow(np.ma.array(u_gt[..., 0], mask=mask), cmap="RdBu_r", vmin=-vx, vmax=vx)
    axes[1, 0].set_title(f"GT u_x")
    for j, route in enumerate(route_order):
        if route in results:
            u, mae, _ = results[route]
            axes[1, j + 1].imshow(np.ma.array(u[..., 0], mask=mask), cmap="RdBu_r", vmin=-vx, vmax=vx)
            axes[1, j + 1].set_title(f"Route {route} u_x\nMAE={mae:.4f}")
    # Error col
    if "A" in results:
        u_a = results["A"][0]
        err_x = np.ma.array(u_a[..., 0] - u_gt[..., 0], mask=mask)
        ve_x = max(abs(err_x).max(), 0.1)
        axes[1, 4].imshow(err_x, cmap="RdBu_r", vmin=-ve_x, vmax=ve_x)
        axes[1, 4].set_title(f"Error u_x (A)")

    # Row 2: u_y
    axes[2, 0].imshow(np.ma.array(u_gt[..., 1], mask=mask), cmap="RdBu_r", vmin=-vy, vmax=vy)
    axes[2, 0].set_title(f"GT u_y")
    for j, route in enumerate(route_order):
        if route in results:
            u, mae, _ = results[route]
            axes[2, j + 1].imshow(np.ma.array(u[..., 1], mask=mask), cmap="RdBu_r", vmin=-vy, vmax=vy)
            axes[2, j + 1].set_title(f"Route {route} u_y\nMAE={mae:.4f}")
    if "A" in results:
        u_a = results["A"][0]
        err_y = np.ma.array(u_a[..., 1] - u_gt[..., 1], mask=mask)
        ve_y = max(abs(err_y).max(), 0.1)
        axes[2, 4].imshow(err_y, cmap="RdBu_r", vmin=-ve_y, vmax=ve_y)
        axes[2, 4].set_title(f"Error u_y (A)")

    # Summary text
    lines = []
    for route in route_order:
        if route in results:
            _, mae, mse = results[route]
            zero = (u_gt[roi] ** 2).sum() / roi.sum()
            ratio = mse / zero if zero > 0 else float("inf")
            lines.append(f"Route {route}: MAE={mae:.4f}  MSE={mse:.6f}  ratio={ratio:.4f}")
    fig.text(0.5, 0.01, "  |  ".join(lines), ha="center", fontsize=9, family="monospace")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {out}")
    plt.close()
