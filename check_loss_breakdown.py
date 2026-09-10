#!/usr/bin/env python3
import torch
import sys
import os

sys.path.append("/data/personal_data/zyy/point-diffusion-cloud")
from models.vae_gaussian_bd import GaussianVAE
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

class DummyArgs:
    def __init__(self, ckpt_args):
        for k, v in vars(ckpt_args).items():
            setattr(self, k, v)
        self.device = 'cpu'

# Load checkpoints
ckpt1_path = "/data/personal_data/zyy/point-diffusion-cloud/logs_bd/BD_2026_06_28__02_01_59_Clean_Finetune_KL/ckpt_0.000000_10000.pt"
ckpt2_path = "/data/personal_data/zyy/point-diffusion-cloud/logs_bd/BD_2026_06_28__02_17_36_Backdoor_Finetune_Stage2/ckpt_0.000000_50000.pt"

dset = ShapeNetCore(
    path="/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k.h5",
    cates=['chair'],
    split='train',
    scale_mode='shape_unit'
)
loader = DataLoader(dset, batch_size=8, shuffle=False)
batch = next(iter(loader))
x = batch['pointcloud']

# Force Normalize x
x = x - x.mean(dim=1, keepdim=True)
max_val = x.abs().max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
x = x / (max_val + 1e-8)

for name, path in [("Phase 1 (10k steps)", ckpt1_path), ("Phase 2 (50k steps)", ckpt2_path)]:
    if not os.path.exists(path):
        print(f"{name} not found")
        continue
    ckpt = torch.load(path, map_location='cpu')
    model = GaussianVAE(ckpt['args'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    with torch.no_grad():
        z_mu, z_sigma = model.encoder(x)
        z = z_mu # using mean
        
        # Calculate KL
        from models.vae_gaussian_bd import standard_normal_logprob, gaussian_entropy
        log_pz = standard_normal_logprob(z).sum(dim=1)
        entropy = gaussian_entropy(logvar=z_sigma)
        loss_prior = (- log_pz - entropy).mean()
        
        # Calculate recons
        loss_recons = model.diffusion.get_loss(x, z)
        
        print(f"\n=== {name} ===")
        print(f"  KL prior loss (unweighted): {loss_prior.item():.4f}")
        print(f"  Weighted KL (kl_weight={ckpt['args'].kl_weight}): {ckpt['args'].kl_weight * loss_prior.item():.4f}")
        print(f"  Reconstruction loss (recons): {loss_recons.item():.4f}")
        print(f"  z_mu mean/std: {z_mu.mean().item():.4f} / {z_mu.std().item():.4f}")
        print(f"  z_sigma (logvar) mean: {z_sigma.mean().item():.4f}")
