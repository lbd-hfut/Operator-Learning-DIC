"""Test image generation with FEM-based displacement fields.

Pipeline:
  1. Load speckle image from test/speckle.png (resize to target if needed)
  2. Generate ROI mask (circle / ring / notch)
  3. Generate FEM displacement field
  4. Forward-warp ROI-masked reference to create target image
  5. Compute valid ROI
  6. Save: ref.png, tar.png, u_field.npy, roi_mask.png
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

# Ensure project root is on sys.path for dataset imports
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataset.warp import warp_image
from dataset.roi import compute_valid_mask

from .roi_shapes import get_roi
from .fem_displacement import generate_displacement

# Default speckle source image
DEFAULT_SPECKLE_IMAGE = Path(__file__).resolve().parent / "speckle.png"


def _load_speckle(image_size, speckle_source=None):
    """Load speckle pattern image, resizing if needed.

    Args:
        image_size: (H, W) target size
        speckle_source: path to speckle image file (default: test/speckle.png)

    Returns:
        [H, W] float32 array in [0, 1]
    """
    src_path = Path(speckle_source) if speckle_source else DEFAULT_SPECKLE_IMAGE
    if not src_path.exists():
        raise FileNotFoundError(
            f"Speckle image not found: {src_path}\n"
            f"Place a speckle pattern at test/speckle.png or specify --speckle_source."
        )

    img = np.array(Image.open(src_path).convert("L"), dtype=np.float32) / 255.0
    h, w = img.shape
    H, W = image_size

    if h != H or w != W:
        img = zoom(img, (H / h, W / w), order=1)

    return img.astype(np.float32)


def generate_test_case(
    case,
    output_dir,
    image_size=(256, 256),
    disp_range=(-4, 4),
    seed=0,
    speckle_source=None,
):
    """Generate a single test case.

    Args:
        case: 'circle', 'ring', or 'notch'
        output_dir: path to save outputs
        image_size: (H, W) of generated images
        disp_range: (min, max) displacement range in pixels
        seed: random seed
        speckle_source: path to speckle image (default: test/speckle.png)

    Returns:
        dict with paths to saved files
    """
    H, W = image_size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load speckle image
    ref = _load_speckle((H, W), speckle_source)

    # 2. Generate ROI mask
    roi = get_roi(case, (H, W))

    # 3. Generate FEM displacement field
    u_field = generate_displacement(case, roi, disp_range=disp_range)

    # 4. Apply ROI mask to reference
    ref_masked = ref.copy()
    ref_masked[~roi] = 0.0

    # 5. Forward-warp to create target image
    tar = warp_image(ref_masked, u_field)

    # 6. Compute valid ROI (pixels that stay within bounds after warp)
    valid_roi = compute_valid_mask(u_field) & roi

    # 7. Save outputs
    ref_path = output_dir / "ref.png"
    tar_path = output_dir / "tar.png"
    u_path = output_dir / "u_field.npy"
    roi_path = output_dir / "roi_mask.png"

    Image.fromarray((ref_masked * 255).astype(np.uint8)).save(ref_path)
    Image.fromarray((np.clip(tar, 0, 1) * 255).astype(np.uint8)).save(tar_path)
    np.save(u_path, u_field)
    Image.fromarray((valid_roi * 255).astype(np.uint8)).save(roi_path)

    # Print summary
    u_mag = np.sqrt(u_field[..., 0] ** 2 + u_field[..., 1] ** 2)
    print(f"  [{case}] ROI: {valid_roi.sum()} px  "
          f"u_x: [{u_field[..., 0][valid_roi].min():.2f}, {u_field[..., 0][valid_roi].max():.2f}]  "
          f"u_y: [{u_field[..., 1][valid_roi].min():.2f}, {u_field[..., 1][valid_roi].max():.2f}]  "
          f"|u|_max: {u_mag[valid_roi].max():.2f}")

    return {
        "ref": ref_path,
        "tar": tar_path,
        "u_field": u_path,
        "roi_mask": roi_path,
        "ref_array": ref_masked,
        "tar_array": tar,
        "u_field_array": u_field,
        "roi_array": valid_roi,
    }
