"""FEM-like displacement field generation via 2D linear elasticity.

Solves the Navier-Cauchy equations (plane stress) on the ROI domain
using finite differences with SOR (Successive Over-Relaxation) iteration.

Provides physically realistic displacement fields for:
  - circle:  diametral compression (Brazilian disk)
  - ring:    diametral compression with inner stress concentration
  - notch:   uniaxial tension with central hole
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def _lame_parameters(E=1.0, nu=0.35, plane_stress=True):
    """Compute Lame constants.

    Args:
        E: Young's modulus
        nu: Poisson's ratio
        plane_stress: if True, use plane stress; otherwise plane strain

    Returns:
        mu, lam: Lame parameters
    """
    mu = E / (2 * (1 + nu))
    if plane_stress:
        lam = E * nu / (1 - nu ** 2)
    else:
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lam


def solve_navier_cauchy(mask, bc_dict, E=1.0, nu=0.35,
                        n_iter=3000, h=1.0):
    """Solve 2D Navier-Cauchy equations (plane stress) on the ROI domain.

    Uses vectorized Jacobi iteration with ghost-node boundary conditions
    (zero normal derivative on free boundaries).

    Args:
        mask: [H, W] bool array, True = inside ROI
        bc_dict: dict with 'ux_nodes', 'uy_nodes', 'ux_vals', 'uy_vals'
        E: Young's modulus
        nu: Poisson's ratio
        n_iter: Jacobi iterations
        h: grid spacing

    Returns:
        u_field: [H, W, 2] displacement field
    """
    H, W = mask.shape
    mu, lam = _lame_parameters(E, nu, plane_stress=True)

    # Weight coefficients
    wu_x = lam + 2 * mu   # u from ∂²u/∂x² in x-equation
    wu_y = mu             # u from ∂²u/∂y² in x-equation
    wv_x = mu             # v from ∂²v/∂x² in y-equation
    wv_y = lam + 2 * mu   # v from ∂²v/∂y² in y-equation
    cross_w = lam + mu    # cross-derivative weight
    denom = 2 * lam + 6 * mu  # common denominator

    # Pad arrays with 1-pixel border for ghost-node boundary conditions
    Ph, Pw = H + 2, W + 2
    u = np.zeros((Ph, Pw), dtype=np.float64)
    v = np.zeros((Ph, Pw), dtype=np.float64)

    # Padded mask
    mask_pad = np.zeros((Ph, Pw), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask

    # Dirichlet BC arrays (padded)
    ux_fixed = np.zeros((Ph, Pw), dtype=bool)
    uy_fixed = np.zeros((Ph, Pw), dtype=bool)

    for (i, j), val in zip(bc_dict.get("ux_nodes", []), bc_dict.get("ux_vals", [])):
        if 0 <= i < H and 0 <= j < W and mask[i, j]:
            u[i + 1, j + 1] = val
            ux_fixed[i + 1, j + 1] = True
    for (i, j), val in zip(bc_dict.get("uy_nodes", []), bc_dict.get("uy_vals", [])):
        if 0 <= i < H and 0 <= j < W and mask[i, j]:
            v[i + 1, j + 1] = val
            uy_fixed[i + 1, j + 1] = True

    # Update masks (interior = ROI minus Dirichlet)
    update_u = mask_pad & (~ux_fixed)
    update_v = mask_pad & (~uy_fixed)

    # Corner validity for cross-derivative terms on interior points.
    # For interior point (i,j), the 4 corners are at (i±1, j±1).
    # With ghost-node filling, all corners are well-defined; we use the
    # mask to drop cross-terms where corners fall outside the physical ROI.
    # corners_ok[i,j] for padded-index (i,j): all 4 corners in mask_pad.
    corners_ok = np.zeros((Ph, Pw), dtype=bool)
    corners_ok[2:-2, 2:-2] = (
        mask_pad[1:-3, 1:-3] & mask_pad[1:-3, 3:-1] &
        mask_pad[3:-1, 1:-3] & mask_pad[3:-1, 3:-1]
    )
    corner_interior = corners_ok[1:-1, 1:-1]  # shape (H, W)

    for it in range(n_iter):
        # Fill ghost cells: copy interior boundary values to exterior
        # (approximates zero-stress Neumann BC: ∂u/∂n = 0)
        # For non-ROI pixels adjacent to ROI: ghost ≈ boundary value
        # For far exterior: value doesn't matter for ROI interior computation

        # Fill edges from interior
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]

        # ---- Neighbor slices (interior only, indices 1..H, 1..W) ----
        u_left = u[1:-1, :-2]
        u_right = u[1:-1, 2:]
        u_up = u[:-2, 1:-1]
        u_down = u[2:, 1:-1]

        v_left = v[1:-1, :-2]
        v_right = v[1:-1, 2:]
        v_up = v[:-2, 1:-1]
        v_down = v[2:, 1:-1]

        # Corner slices for cross-derivatives
        v_nw = v[:-2, :-2]
        v_ne = v[:-2, 2:]
        v_sw = v[2:, :-2]
        v_se = v[2:, 2:]
        u_nw = u[:-2, :-2]
        u_ne = u[:-2, 2:]
        u_sw = u[2:, :-2]
        u_se = u[2:, 2:]

        # Cross-derivative terms (zero where corners incomplete)
        cross_v = np.where(corner_interior,
                           (v_nw + v_se - v_ne - v_sw) / 4.0, 0.0)
        cross_u = np.where(corner_interior,
                           (u_nw + u_se - u_ne - u_sw) / 4.0, 0.0)

        # Jacobi update
        u_new = (wu_x * (u_left + u_right) + wu_y * (u_up + u_down)
                 + cross_w * cross_v) / denom
        v_new = (wv_x * (v_left + v_right) + wv_y * (v_up + v_down)
                 + cross_w * cross_u) / denom

        # Apply updates only where allowed (interior non-Dirichlet)
        u_core = u[1:-1, 1:-1]
        v_core = v[1:-1, 1:-1]
        u[1:-1, 1:-1] = np.where(update_u[1:-1, 1:-1], u_new, u_core)
        v[1:-1, 1:-1] = np.where(update_v[1:-1, 1:-1], v_new, v_core)

    # Return unpadded
    return np.stack([u[1:-1, 1:-1], v[1:-1, 1:-1]], axis=-1).astype(np.float32)


def _find_boundary_nodes(mask, side="top", frac=0.15, angular_range=None, center=None):
    """Find boundary nodes on a specified side of the ROI.

    Boundary nodes are mask pixels adjacent to at least one non-mask pixel.

    Args:
        mask: [H, W] bool array
        side: 'top', 'bottom', 'left', 'right'
        frac: fraction of the range to consider as the loading zone
        angular_range: (min_angle, max_angle) in radians for curved ROIs
        center: (cy, cx) for angular filtering

    Returns:
        list of (i, j) tuples
    """
    H, W = mask.shape

    # Find boundary: mask pixels with at least one non-mask neighbor
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    by, bx = np.where(boundary)

    if len(by) == 0:
        return []

    if side == "top":
        subset = by < (by.min() + frac * H)
    elif side == "bottom":
        subset = by > (by.max() - frac * H)
    elif side == "left":
        subset = bx < (bx.min() + frac * W)
    elif side == "right":
        subset = bx > (bx.max() - frac * W)
    else:
        raise ValueError(f"Unknown side: {side}")

    # Angular filtering for curved geometries
    if angular_range is not None and center is not None:
        cy, cx = center
        angles = np.arctan2(by - cy, bx - cx)
        ang_min, ang_max = angular_range
        # Normalize angles to [-pi, pi]
        in_angle = (angles >= ang_min) & (angles <= ang_max)
        subset = subset & in_angle

    nodes = [(by[k], bx[k]) for k in range(len(by)) if subset[k]]
    return nodes


def generate_displacement(case, mask, disp_range=(-4, 4),
                          E=1.0, nu=0.35, n_iter=4000):
    """Generate FEM-like displacement field for a given case.

    Boundary conditions match realistic experimental setups:
      - circle/ring (compression):
          Top vertex:   u_y = prescribed (compression), u_x = 0
          Bottom vertex: u_x = u_y = 0  (fixed support)
      - notch (tension):
          Top edge:     u_y = prescribed (tension), u_x = 0
          Bottom edge:  u_x = u_y = 0  (fixed support)

    Args:
        case: 'circle', 'ring', or 'notch'
        mask: [H, W] bool array, ROI mask
        disp_range: (min_val, max_val) target displacement range in pixels
        E: Young's modulus
        nu: Poisson's ratio
        n_iter: Jacobi iterations

    Returns:
        u_field: [H, W, 2] displacement field in pixels
    """
    H, W = mask.shape
    cy, cx = H / 2, W / 2

    bc_dict = {"ux_nodes": [], "uy_nodes": [], "ux_vals": [], "uy_vals": []}

    def _add_ux(i, j, val):
        bc_dict["ux_nodes"].append((i, j))
        bc_dict["ux_vals"].append(val)

    def _add_uy(i, j, val):
        bc_dict["uy_nodes"].append((i, j))
        bc_dict["uy_vals"].append(val)

    if case in ("circle", "ring"):
        # ── Compression: top vertex loaded, bottom vertex fixed ──
        # Top loading point: narrow angular sector at θ ≈ -π/2 (top)
        top_nodes = _find_boundary_nodes(
            mask, side="top", frac=0.08,
            angular_range=(-2.0, -1.2), center=(cy, cx),
        )
        # Bottom fixed point: narrow angular sector at θ ≈ +π/2 (bottom)
        bottom_nodes = _find_boundary_nodes(
            mask, side="bottom", frac=0.08,
            angular_range=(1.2, 2.0), center=(cy, cx),
        )

        for (i, j) in top_nodes:
            _add_ux(i, j, 0.0)    # u_x = 0 (constrained)
            _add_uy(i, j, -1.0)   # u_y = -1 (compression downward)

        for (i, j) in bottom_nodes:
            _add_ux(i, j, 0.0)    # u_x = 0 (fixed)
            _add_uy(i, j, 0.0)    # u_y = 0 (fixed)

    elif case == "notch":
        # ── Notch tension: top edge loaded upward, bottom edge fixed ──
        boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        by, bx = np.where(boundary)
        if len(by) > 0:
            by_min, by_max = by.min(), by.max()
            top_mask = by < (by_min + 5)
            top_nodes = [(by[k], bx[k]) for k in range(len(by)) if top_mask[k]]
            bottom_mask = by > (by_max - 5)
            bottom_nodes = [(by[k], bx[k]) for k in range(len(by)) if bottom_mask[k]]
        else:
            top_nodes, bottom_nodes = [], []

        for (i, j) in top_nodes:
            _add_ux(i, j, 0.0)    # u_x = 0 (constrained)
            _add_uy(i, j, 1.0)    # u_y = +1 (tension upward)

        for (i, j) in bottom_nodes:
            _add_ux(i, j, 0.0)    # u_x = 0 (fixed)
            _add_uy(i, j, 0.0)    # u_y = 0 (fixed)

    elif case == "full":
        # ── Full-plate compression: top edge loaded downward, bottom edge fixed ──
        boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        by, bx = np.where(boundary)
        if len(by) > 0:
            by_min, by_max = by.min(), by.max()
            top_mask = by < (by_min + 5)
            top_nodes = [(by[k], bx[k]) for k in range(len(by)) if top_mask[k]]
            bottom_mask = by > (by_max - 5)
            bottom_nodes = [(by[k], bx[k]) for k in range(len(by)) if bottom_mask[k]]
        else:
            top_nodes, bottom_nodes = [], []

        for (i, j) in top_nodes:
            _add_ux(i, j, 0.0)    # u_x = 0 (constrained)
            _add_uy(i, j, -1.0)   # u_y = -1 (compression downward)

        for (i, j) in bottom_nodes:
            _add_ux(i, j, 0.0)    # u_x = 0 (fixed)
            _add_uy(i, j, 0.0)    # u_y = 0 (fixed)

    else:
        raise ValueError(f"Unknown case: {case}")

    if len(bc_dict["ux_nodes"]) == 0 and len(bc_dict["uy_nodes"]) == 0:
        raise RuntimeError(f"No Dirichlet BC nodes found for case '{case}'. Check ROI mask.")

    # Solve
    u_field = solve_navier_cauchy(mask, bc_dict, E=E, nu=nu, n_iter=n_iter)

    # Scale to target range with tunable u_x/u_y ratio
    u_field = _scale_displacement(u_field, mask, disp_range)

    return u_field


def _scale_displacement(u_field, mask, disp_range):
    """Scale displacement field so that:

    1. The loading component (u_y for circle/ring, u_y for notch) spans
       the target range.
    2. The transverse component (u_x) is scaled independently to achieve
       a range of ~0.5× the loading component range.

    This keeps the FEM spatial structure while adjusting magnitudes
    to match realistic experimental deformation patterns.

    Args:
        u_field: [H, W, 2] raw displacement from FEM solver
        mask: [H, W] bool, ROI mask
        disp_range: (min_val, max_val) target range for loading component

    Returns:
        scaled u_field [H, W, 2]
    """
    target_max = max(abs(disp_range[0]), abs(disp_range[1]))

    # Identify loading component: the one with larger range
    ux_range = u_field[..., 0][mask].max() - u_field[..., 0][mask].min()
    uy_range = u_field[..., 1][mask].max() - u_field[..., 1][mask].min()

    if uy_range >= ux_range:
        # u_y is the loading direction (circle/ring/notch)
        if uy_range > 1e-10:
            u_field[..., 1] *= target_max / uy_range
        # Scale u_x to ~half of u_y range
        ux_new_range = u_field[..., 0][mask].max() - u_field[..., 0][mask].min()
        if ux_new_range > 1e-10:
            target_ux = target_max * 0.5
            u_field[..., 0] *= target_ux / ux_new_range
    else:
        # u_x is the loading direction
        if ux_range > 1e-10:
            u_field[..., 0] *= target_max / ux_range
        uy_new_range = u_field[..., 1][mask].max() - u_field[..., 1][mask].min()
        if uy_new_range > 1e-10:
            target_uy = target_max * 0.5
            u_field[..., 1] *= target_uy / uy_new_range

    return u_field.astype(np.float32)
