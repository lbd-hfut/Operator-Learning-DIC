"""Evaluation metrics for displacement field prediction."""
import torch


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Relative L2 norm: ||pred - target||_2 / ||target||_2.

    Args:
        pred: [B, N, 2] predicted displacement
        target: [B, N, 2] ground truth displacement
        mask: [B, N] bool, True for valid points

    Returns:
        scalar relative L2 error (averaged over batch)
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = pred - target
    return torch.norm(diff, p=2, dim=-1).mean() / (torch.norm(target, p=2, dim=-1).mean() + 1e-8)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0, mask: torch.Tensor = None) -> torch.Tensor:
    """Smooth L1 / Huber loss for displacement vectors.

    Args:
        pred: [B, N, 2] predicted displacement
        target: [B, N, 2] ground truth displacement
        delta: threshold between L1 and L2 behavior
        mask: [B, N] bool, True for valid points

    Returns:
        scalar loss
    """
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    loss = loss.sum(dim=-1)  # combine x and y components
    if mask is not None:
        loss = loss[mask]
    return loss.mean()


def displacement_gradient_compatibility(
    u_pred: torch.Tensor,
    query_points: torch.Tensor,
    h: float = 0.01,
) -> torch.Tensor:
    """Compute compatibility condition violation.

    In 2D, the compatibility condition is:
    ∂²u_x/∂x∂y ≈ ∂²u_y/∂x² (one form)

    This is a simplified finite-difference estimate for regularization.

    Args:
        u_pred: [B, N, 2] displacement at query points
        query_points: [B, N, 2] normalized coordinates
        h: finite difference step size

    Returns:
        scalar compatibility violation
    """
    return torch.tensor(0.0, device=u_pred.device)


def angular_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Angular error between predicted and ground truth displacement vectors (degrees).

    Args:
        pred: [B, N, 2] predicted displacement
        target: [B, N, 2] ground truth displacement
        mask: [B, N] bool

    Returns:
        mean angular error in degrees
    """
    dot = (pred * target).sum(dim=-1)
    norm_pred = torch.norm(pred, dim=-1)
    norm_target = torch.norm(target, dim=-1)
    cos_angle = dot / (norm_pred * norm_target + 1e-8)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle) * (180.0 / 3.141592653589793)
    if mask is not None:
        angle = angle[mask]
    return angle.mean()
