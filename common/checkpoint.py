"""Checkpoint save/load utilities with _last / _best naming.

_last.pt: always overwritten with the latest state (for resume)
_best.pt: saved only when the tracked metric improves
"""
import os
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    current_lr: float,
    checkpoint_dir: str,
    experiment_name: str,
    is_best: bool = False,
):
    """Save training checkpoint.

    Always overwrites <name>_last.pt. Also saves <name>_best.pt when is_best=True.

    Args:
        model: model with weights to save
        optimizer: optimizer state
        scheduler: lr scheduler state
        epoch: current epoch
        global_step: current global step
        best_metric: best validation/training metric so far
        current_lr: current learning rate
        checkpoint_dir: directory to save to
        experiment_name: name prefix for checkpoint files
        is_best: if True, also saves <experiment_name>_best.pt
    """
    ensure_dir(checkpoint_dir)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "lr": current_lr,
    }

    last_path = os.path.join(checkpoint_dir, f"{experiment_name}_last.pt")
    torch.save(checkpoint, last_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pt")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    device: str = "cuda",
) -> dict:
    """Load a training checkpoint and restore model/optimizer/scheduler state.

    Args:
        path: path to checkpoint file
        model: model to load weights into
        optimizer: optional optimizer to restore state
        scheduler: optional scheduler to restore state
        device: device to map tensors to

    Returns:
        dict with epoch, global_step, best_metric, lr
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "lr": checkpoint.get("lr", None),
    }
