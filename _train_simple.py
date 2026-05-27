"""Train full model on 800 samples with constant lr=1e-3, no warmup."""
import sys; sys.path.insert(0, ".")
import torch; import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset.folder_dataset import FolderDICDataset
from dataset.collate import collate_fn
from deformation_inverse_operator.model import InverseOperatorModel
from deformation_inverse_operator.config import InverseOperatorConfig

device = torch.device("cuda")
dataset = FolderDICDataset("dataset/dataset/2026-05-26/train")
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

config = InverseOperatorConfig()
config.warmup_steps = 0
config.fourier_scale = 10.0
model = InverseOperatorModel(config).to(device)
opt = AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model.train()
step = 0
while step < 5000:
    for batch in loader:
        if step >= 5000:
            break
        ref = batch["ref_img"].to(device); tar = batch["tar_img"].to(device)
        qpts = batch["query_points"].to(device); u_gt = batch["u_gt"].to(device)
        qmask = batch["query_mask"].to(device)

        opt.zero_grad()
        u_pred = model(ref, tar, qpts)
        loss = criterion(u_pred[qmask], u_gt[qmask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            mae = (u_pred[qmask] - u_gt[qmask]).abs().mean().item()
            print(f"step {step:5d}: MSE={loss.item():.6f}, MAE={mae:.4f}px, "
                  f"pred=[{u_pred.min():.2f},{u_pred.max():.2f}], gt=[{u_gt.min():.2f},{u_gt.max():.2f}]")
        step += 1
