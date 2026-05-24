"""Checkpoint save/load utilities."""
import os
import torch


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    path: str,
    is_best: bool = False,
):
    """Save training checkpoint.

    Args:
        model: model with weights to save
        optimizer: optimizer state
        scheduler: learning rate scheduler state
        epoch: current epoch
        global_step: current global step
        best_metric: best validation metric value
        path: file path to save to
        is_best: if True, also saves a 'best.pt' copy in the same directory
    """
    ensure_dir(os.path.dirname(path))
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best.pt")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    device: str = "cuda",
):
    """Load training checkpoint.

    Returns:
        epoch, global_step, best_metric
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("global_step", 0),
        checkpoint.get("best_metric", float("inf")),
    )
