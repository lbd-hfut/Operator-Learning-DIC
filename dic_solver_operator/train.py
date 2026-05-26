"""Training script for DIC Solver Operator (Route A).

Usage:
    # On-the-fly synthetic data
    python -m dic_solver_operator.train

    # YAML config (recommended)
    python -m dic_solver_operator.train --config config/training.yaml

    # Resume from checkpoint
    python -m dic_solver_operator.train --resume checkpoints/dic_operator_last.pt

    # Multi-GPU via DDP
    torchrun --nproc_per_node=4 -m dic_solver_operator.train --use_ddp --config config/training.yaml
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.config import DatasetConfig
from dataset.dic_dataset import DICDataset
from dataset.folder_dataset import FolderDICDataset
from dataset.collate import collate_fn
from dic_solver_operator.config import SolverOperatorConfig
from dic_solver_operator.model import SolverOperatorModel
from common.losses import CompositeLoss
from common.checkpoint import save_checkpoint, ensure_dir
from common.config_utils import load_yaml, apply_yaml_overrides


def train(rank: int, world_size: int, config: SolverOperatorConfig,
          data_config: DatasetConfig = None, dataset_dir: str = None,
          resume_path: str = None):
    """Main training loop."""

    # --- Distributed setup ---
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    if dataset_dir:
        dataset = FolderDICDataset(split_dir=dataset_dir)
    else:
        dataset = DICDataset(data_config)
    sampler = DistributedSampler(dataset) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=0 if os.name == "nt" else 4,
        pin_memory=True,
    )

    # --- Model ---
    model = SolverOperatorModel(config).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # --- Optimizer & scheduler ---
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup = LinearLR(optimizer, start_factor=config.warmup_start_factor, end_factor=1.0,
                      total_iters=config.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=config.max_steps - config.warmup_steps,
                               eta_min=config.learning_rate * 1e-2)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[config.warmup_steps])

    # --- Loss ---
    criterion = CompositeLoss(
        data_loss_type=config.data_loss_type,
        reg_loss_type=config.reg_loss_type,
        lambda_reg=config.lambda_reg,
    )

    # --- Resume or start fresh ---
    ensure_dir(config.checkpoint_dir)
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    resume_state = None

    if resume_path:
        model_to_load = model.module if is_distributed else model
        resume_state = load_checkpoint(resume_path, model_to_load, optimizer, scheduler, device)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        best_metric = resume_state["best_metric"]
        if rank == 0:
            print(f"Resumed from {resume_path} (step={global_step}, best={best_metric:.6f})")

    # --- Training loop ---
    model.train()
    for epoch in range(start_epoch, config.max_steps // len(loader) + 1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for batch in loader:
            # Skip steps already done when resuming
            if resume_state is not None and global_step < resume_state["global_step"]:
                global_step += 1
                continue

            ref = batch["ref_img"].to(device)
            tar = batch["tar_img"].to(device)
            qpts = batch["query_points"].to(device)
            u_gt = batch["u_gt"].to(device)
            qmask = batch["query_mask"].to(device)

            # Forward
            u_pred = model(ref, tar, qpts) if not is_distributed else model.module(ref, tar, qpts)

            # Loss
            loss_dict = criterion(u_pred, u_gt, qpts, qmask)
            loss = loss_dict["loss"]

            # Backward
            loss.backward()
            if (global_step + 1) % config.accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Logging
            if rank == 0 and global_step % config.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[Step {global_step:06d}] "
                    f"loss={loss.item():.6f} "
                    f"data_loss={loss_dict['data_loss'].item():.6f} "
                    f"lr={current_lr:.2e}"
                )

            # Checkpoint
            if rank == 0 and global_step % config.save_every == 0 and global_step > 0:
                current_lr = scheduler.get_last_lr()[0]
                current_loss = loss_dict["data_loss"].item()
                is_best = current_loss < best_metric
                if is_best:
                    best_metric = current_loss
                    print(f"  -> new best: {best_metric:.6f}")

                model_to_save = model.module if is_distributed else model
                save_checkpoint(
                    model_to_save, optimizer, scheduler,
                    epoch=epoch, global_step=global_step,
                    best_metric=best_metric, current_lr=current_lr,
                    checkpoint_dir=config.checkpoint_dir,
                    experiment_name=config.experiment_name,
                    is_best=is_best,
                )

            global_step += 1
            if global_step >= config.max_steps:
                break

        if global_step >= config.max_steps:
            break

    # Save final checkpoint
    if rank == 0:
        current_lr = scheduler.get_last_lr()[0]
        model_to_save = model.module if is_distributed else model
        save_checkpoint(
            model_to_save, optimizer, scheduler,
            epoch=epoch, global_step=global_step,
            best_metric=best_metric, current_lr=current_lr,
            checkpoint_dir=config.checkpoint_dir,
            experiment_name=config.experiment_name,
            is_best=False,
        )
        print(f"Training complete. best_metric={best_metric:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="path to YAML training config")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint to resume from")
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="path to pre-generated folder dataset (overrides YAML)")
    args = parser.parse_args()

    # Load YAML config if provided
    yaml_cfg = {}
    if args.config:
        yaml_cfg = load_yaml(args.config)
        print(f"Loaded config from {args.config}")

    # Build model config
    config = SolverOperatorConfig()
    apply_yaml_overrides(config, yaml_cfg, section="model")
    apply_yaml_overrides(config, yaml_cfg, section="training")
    apply_yaml_overrides(config, yaml_cfg, section="loss")
    apply_yaml_overrides(config, yaml_cfg, section="logging")

    # Resume path: CLI > YAML > auto-detect _last.pt
    resume_path = args.resume or yaml_cfg.get("resume_a_from")
    if not resume_path:
        auto_last = os.path.join(config.checkpoint_dir, f"{config.experiment_name}_last.pt")
        if os.path.exists(auto_last):
            resume_path = auto_last
            print(f"Auto-resume from {auto_last}")

    # Dataset
    dataset_dir = args.dataset_dir or yaml_cfg.get("dataset_dir")
    if dataset_dir:
        from pathlib import Path
        p = Path(dataset_dir)
        if not (p / "ref").exists() and (p / "train" / "ref").exists():
            dataset_dir = str(p / "train")
        data_config = None
    else:
        data_config = DatasetConfig(n_samples=10000, seed=42)

    if args.use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        train(local_rank, world_size, config, data_config, dataset_dir, resume_path)
    else:
        train(0, 1, config, data_config, dataset_dir, resume_path)


if __name__ == "__main__":
    main()
