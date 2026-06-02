"""ROI shape generation for irregular-ROI test cases.

Provides three ROI types mimicking realistic experimental setups:
  - circle:  circular disk (compression test)
  - ring:    concentric ring (compression test, stress concentration at hole)
  - notch:   dogbone plate with central hole (tension test)
"""

import numpy as np


def make_circle(shape, radius=None, center=None):
    """Circular disk ROI.

    Args:
        shape: (H, W) output shape
        radius: circle radius in pixels (default: 0.35 * min(H, W))
        center: (cy, cx) center (default: image center)

    Returns:
        roi: [H, W] bool array, True = inside ROI
    """
    H, W = shape
    if radius is None:
        radius = int(min(H, W) * 0.35)
    if center is None:
        center = (H / 2, W / 2)
    cy, cx = center
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2


def make_ring(shape, r_outer=None, r_inner=None, center=None):
    """Concentric ring ROI (annulus).

    Args:
        shape: (H, W) output shape
        r_outer: outer radius (default: 0.40 * min(H, W))
        r_inner: inner radius (default: 0.14 * min(H, W))
        center: (cy, cx) center

    Returns:
        roi: [H, W] bool array
    """
    H, W = shape
    if r_outer is None:
        r_outer = int(min(H, W) * 0.40)
    if r_inner is None:
        r_inner = int(min(H, W) * 0.14)
    if center is None:
        center = (H / 2, W / 2)
    cy, cx = center
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (d2 < r_outer ** 2) & (d2 > r_inner ** 2)


def make_notch(shape, margin=None, notch_width=None, hole_radius=None):
    """Dogbone-shaped ROI with central hole (notch plate).

    The shape is a rectangular plate with narrowed waist (removed side
    material) and a central circular hole — mimicking a typical tensile
    test specimen with stress concentration.

    Args:
        shape: (H, W) output shape
        margin: vertical margin in pixels (default: 5% of H)
        notch_width: width of the narrow (waist) section (default: 35% of W)
        hole_radius: radius of central hole (default: 8% of min(H, W))

    Returns:
        roi: [H, W] bool array
    """
    H, W = shape
    if margin is None:
        margin = int(H * 0.05)
    if notch_width is None:
        notch_width = int(W * 0.35)
    if hole_radius is None:
        hole_radius = int(min(H, W) * 0.08)

    roi = np.ones(shape, dtype=bool)

    # Remove vertical margins
    roi[:margin, :] = False
    roi[-margin:, :] = False

    # Remove side material to create the narrow waist
    roi[:, :notch_width] = False
    roi[:, -notch_width:] = False

    # Central hole
    cy, cx = H / 2, W / 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    hole = (yy - cy) ** 2 + (xx - cx) ** 2 < hole_radius ** 2
    roi = roi & ~hole

    return roi


def make_full(shape):
    """Full-image ROI (no masking).

    Args:
        shape: (H, W) output shape

    Returns:
        roi: [H, W] bool array, all True
    """
    return np.ones(shape, dtype=bool)


# Mapping from case name to generator and its default kwargs
ROI_GENERATORS = {
    "circle": (make_circle, {"radius": None, "center": None}),
    "ring": (make_ring, {"r_outer": None, "r_inner": None, "center": None}),
    "notch": (make_notch, {"margin": None, "notch_width": None, "hole_radius": None}),
    "full": (make_full, {}),
}


def get_roi(case: str, shape, **kwargs):
    """Generate ROI mask for a named case.

    Args:
        case: one of 'circle', 'ring', 'notch'
        shape: (H, W) tuple
        **kwargs: forwarded to the generator

    Returns:
        roi: [H, W] bool array
    """
    if case not in ROI_GENERATORS:
        raise ValueError(f"Unknown case '{case}'. Choose from {list(ROI_GENERATORS.keys())}.")
    gen_fn, defaults = ROI_GENERATORS[case]
    params = {**defaults, **{k: v for k, v in kwargs.items() if v is not None}}
    return gen_fn(shape, **params)
