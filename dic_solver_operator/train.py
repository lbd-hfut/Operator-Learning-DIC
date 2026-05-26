"""Training script for DIC Solver Operator (Route A).

Usage:
    # On-the-fly synthetic data
    python -m dic_solver_operator.train

    # Pre-generated folder dataset
    python -m dic_solver_operator.train --dataset_dir dataset/dataset/2026-05-26/train

    # Multi-GPU via DDP
    torchrun --nproc_per_node=4 -m dic_solver_operator.train --use_ddp --dataset_dir ...
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
from torch.optim.lr_scheduler import CosineAnnealingLR

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


def train(rank: int, world_size: int, config: SolverOperatorConfig,
          data_config: DatasetConfig = None, dataset_dir: str = None):
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
        num_workers=4,
        pin_memory=True,
    )

    # --- Model ---
    model = SolverOperatorModel(config).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # --- Optimizer & scheduler ---
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_steps)

    # --- Loss ---
    criterion = CompositeLoss(
        data_loss_type=config.data_loss_type,
        reg_loss_type=config.reg_loss_type,
        lambda_reg=config.lambda_reg,
    )

    # --- Training loop ---
    ensure_dir(config.checkpoint_dir)
    global_step = 0
    best_metric = float("inf")

    model.train()
    for epoch in range(config.max_steps // len(loader) + 1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for batch in loader:
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
            if global_step % config.accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Logging
            if rank == 0 and global_step % config.log_every == 0:
                print(
                    f"[Step {global_step:06d}] "
                    f"loss={loss.item():.6f} "
                    f"data_loss={loss_dict['data_loss'].item():.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            # Checkpoint
            if rank == 0 and global_step % config.save_every == 0 and global_step > 0:
                ckpt_path = os.path.join(config.checkpoint_dir, f"step_{global_step:06d}.pt")
                model_to_save = model.module if is_distributed else model
                save_checkpoint(
                    model_to_save, optimizer, scheduler, epoch, global_step, best_metric, ckpt_path
                )

            global_step += 1
            if global_step >= config.max_steps:
                break

        if global_step >= config.max_steps:
            break

    if rank == 0:
        print("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="path to pre-generated folder dataset (bypasses on-the-fly generation)")
    args = parser.parse_args()

    config = SolverOperatorConfig()

    if args.dataset_dir:
        data_config = None
    else:
        data_config = DatasetConfig(n_samples=10000, seed=42)

    if args.use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        train(local_rank, world_size, config, data_config, args.dataset_dir)
    else:
        train(0, 1, config, data_config, args.dataset_dir)


if __name__ == "__main__":
    main()
