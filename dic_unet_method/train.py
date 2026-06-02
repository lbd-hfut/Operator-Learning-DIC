"""Training script for U-Net DIC Method (Route C).

Usage:
    python -m dic_unet_method.train --config config/training.yaml
    python -m dic_unet_method.train --dataset_dir dataset/dataset/2026-05-27/train
    python -m dic_unet_method.train --resume checkpoints/route_c/last.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.folder_dataset import FolderDICDataset
from dataset.collate import collate_fn
from dic_unet_method.config import UnetDICConfig
from dic_unet_method.model import UnetDICModel
from common.checkpoint import save_checkpoint, ensure_dir
from common.config_utils import load_yaml, apply_yaml_overrides


def train(rank: int, world_size: int, config: UnetDICConfig,
          dataset_dir: str = None, resume_path: str = None):
    """Main training loop for Route C."""

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    dataset = FolderDICDataset(
        dataset_dir,
        n_query_min=4096,
        n_query_max=8192,
    )
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                            num_workers=0, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)

    # ---- Model ----
    model = UnetDICModel(config).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # ---- Optimizer & scheduler ----
    opt = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.MSELoss()

    steps_per_epoch = len(loader)
    total_steps = config.max_epochs * steps_per_epoch
    warmup_steps = config.warmup_steps
    if warmup_steps > 0:
        warmup = LinearLR(opt, start_factor=config.warmup_start_factor, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(opt, T_max=total_steps)

    start_epoch = 0
    global_step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt.get("optimizer_state_dict", opt.state_dict()))
        global_step = ckpt.get("step", 0)
        start_epoch = global_step // steps_per_epoch
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # ---- Training loop ----
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    best_mse = float("inf")
    print(f"Route C | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch, {config.max_epochs} epochs")

    resume_step_in_epoch = global_step % steps_per_epoch if resume_path else 0

    for epoch in range(start_epoch, config.max_epochs):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        step_in_epoch = 0

        for batch in loader:
            if resume_path and epoch == start_epoch and step_in_epoch < resume_step_in_epoch:
                step_in_epoch += 1
                continue

            if global_step >= total_steps:
                break

            ref = batch["ref_img"].to(device)           # [B, 1, H, W]
            tar = batch["tar_img"].to(device)           # [B, 1, H, W]
            u_gt = batch["u_gt"].to(device)             # [B, N_q, 2]
            qmask = batch["query_mask"].to(device)       # [B, N_q]

            opt.zero_grad()

            # UNet predicts dense [B, 2, H, W]; sample at query points for loss
            u_dense = model(ref, tar)                    # [B, 2, H, W]

            # Sample dense prediction at query points
            B, _, H, W = u_dense.shape
            qpts = batch["query_points"].to(device)     # [B, N_q, 2]
            grid = qpts * 2.0 - 1.0                     # [0,1] → [-1,1]
            grid = grid.unsqueeze(2)                     # [B, N_q, 1, 2]
            u_pred = torch.nn.functional.grid_sample(
                u_dense, grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )                                            # [B, 2, N_q, 1]
            u_pred = u_pred.squeeze(-1).transpose(1, 2)  # [B, N_q, 2]

            loss = criterion(u_pred[qmask], u_gt[qmask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            scheduler.step()

            epoch_loss_sum += loss.item()
            epoch_batches += 1

            if step_in_epoch % config.log_every == 0:
                with torch.no_grad():
                    mae = (u_pred[qmask] - u_gt[qmask]).abs().mean().item()
                    zero_mse = (u_gt[qmask] ** 2).mean().item()
                print(f"epoch [{epoch:03d}][{step_in_epoch:04d}] "
                      f"MSE={loss.item():.6f} (zero={zero_mse:.4f}) "
                      f"MAE={mae:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

            if global_step % config.save_every == 0 and global_step > 0:
                if loss.item() < best_mse:
                    best_mse = loss.item()
                    save_checkpoint(ckpt_dir / "best.pt", model, config, opt, global_step)
                    print(f"  -> saved best (MSE={best_mse:.6f})")

            global_step += 1
            step_in_epoch += 1

        # --- End of epoch ---
        if epoch_batches > 0:
            avg_loss = epoch_loss_sum / epoch_batches
            print(f"epoch [{epoch:03d}] avg MSE={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if global_step >= total_steps:
            break

    save_checkpoint(ckpt_dir / "last.pt", model, config, opt, global_step)
    print(f"Done. Final MSE={loss.item():.6f}, best MSE={best_mse:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Route C: U-Net DIC Training")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Folder dataset path (overrides YAML)")
    args = parser.parse_args()

    config = UnetDICConfig()
    if args.config:
        overrides = load_yaml(args.config)
        config = apply_yaml_overrides(config, overrides)

    dataset_dir = args.dataset_dir or "dataset/dataset/2026-05-27/train"

    if args.use_ddp:
        world_size = torch.cuda.device_count()
        import torch.multiprocessing as mp
        mp.spawn(train, args=(world_size, config, dataset_dir, args.resume),
                 nprocs=world_size, join=True)
    else:
        train(0, 1, config, dataset_dir, args.resume)


if __name__ == "__main__":
    main()
