"""Visualization utilities for DIC datasets."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Optional


def plot_speckle_pattern(image: np.ndarray, title: str = "Speckle Pattern"):
    """Plot a speckle image with histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(title)
    ax1.axis("off")
    ax2.hist(image.ravel(), bins=50)
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Count")
    return fig


def plot_deformation_field(
    u_field: np.ndarray,
    step: int = 16,
    title: str = "Displacement Field",
    ax=None,
):
    """Plot displacement field as a quiver plot.

    Args:
        u_field: [H, W, 2] displacement in pixels
        step: subsampling step for arrows
        title: plot title
        ax: optional matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    H, W = u_field.shape[:2]
    y, x = np.mgrid[step // 2 : H : step, step // 2 : W : step]
    u_x = u_field[::step, ::step, 0]
    u_y = u_field[::step, ::step, 1]

    ax.quiver(x, y, u_x, -u_y, angles="xy", scale_units="xy", scale=1)
    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    return ax


def plot_image_pair(
    ref_img: np.ndarray,
    tar_img: np.ndarray,
    title: str = "Reference vs Target",
):
    """Show reference and target images side by side with difference map."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    ax1.imshow(ref_img, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Reference")
    ax1.axis("off")

    ax2.imshow(tar_img, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Target (Warped)")
    ax2.axis("off")

    diff = np.abs(ref_img - tar_img)
    ax3.imshow(diff, cmap="hot")
    ax3.set_title(f"|Diff| (max={diff.max():.3f})")
    ax3.axis("off")

    fig.suptitle(title)
    return fig


def plot_query_points_on_image(
    image: np.ndarray,
    query_points: np.ndarray,
    u_gt: np.ndarray = None,
    title: str = "Query Points",
):
    """Overlay query points and displacement vectors on image."""
    H, W = image.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    px = query_points[:, 0] * (W - 1)
    py = query_points[:, 1] * (H - 1)
    ax.scatter(px, py, c="red", s=5, alpha=0.6, label="Query points")

    if u_gt is not None:
        ax.quiver(
            px, py, u_gt[:, 0], -u_gt[:, 1],
            angles="xy", scale_units="xy", scale=0.5,
            color="cyan", alpha=0.5, width=0.002,
        )

    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.legend()
    return fig
