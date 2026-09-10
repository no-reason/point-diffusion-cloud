import os
import math
import argparse
import torch
import distutils
import distutils.version
import torch.utils.tensorboard
import json
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import numpy as np

from utils.dataset import *
from utils.misc import *
from utils.misc import get_logger, get_new_log_dir
from utils.data import *
from models.vae_gaussian import *
from models.common import reparameterize_gaussian, standard_normal_logprob, gaussian_entropy
from tools.pcd_backdoor_framework import (
    compute_geometric_mask, project_mask_to_latent,
    pgd_latent_optimization, pgd_distribution_latent_optimization,
    pgd_stochastic_distribution_latent_optimization,
    gaussian_kl_divergence, gaussian_kl_to_std_normal,
    compute_instance_kl_loss, compute_mmd_loss
)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
parser.add_argument('--target_file', type=str, default='./targets/stageC8E_fixed_airplane_target.npy')
parser.add_argument('--target_mode', type=str, choices=['single', 'distribution'], default='single')
parser.add_argument('--target_categories', type=str_list, default=['airplane'])

parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--scale_mode', type=str, default='shape_bbox')
parser.add_argument('--train_batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=100000)
parser.add_argument('--sched_end_epoch', type=int, default=200000)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_root', type=str, default='./logs_stageC')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--tag', type=str, default='Manifold_Latent_Backdoor')

# Backdoor specific
parser.add_argument('--poison_rate', type=float, default=0.03125) # 3.125%
parser.add_argument('--eps', type=float, default=0.5)
parser.add_argument('--poison_loss_weight', type=float, default=8.0)
parser.add_argument('--lambda_align', type=float, default=1.0) # Alignment loss weight
parser.add_argument('--align_loss', type=str, choices=['kl_instance', 'mmd'], default='kl_instance', help='Alignment loss function: kl_instance or mmd')
parser.add_argument('--logvar_lower_bound', type=float, default=-10.0, help='Lower bound for delta_logvar clamping during latent optimization')
parser.add_argument('--pgd_steps', type=int, default=100)
parser.add_argument('--pgd_lr', type=float, default=0.01)
parser.add_argument('--lambda_cd', type=float, default=1.0)
parser.add_argument('--kl_weight', type=float, default=0.001)

args = parser.parse_args()
seed_all(args.seed)

log_dir = get_new_log_dir(args.log_root, prefix=args.tag, postfix='')
logger = get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)

logger.info(f"Log dir: {log_dir}")
with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

# Load Target Data
if args.target_mode == 'distribution':
    logger.info(f"Target mode: DISTRIBUTION (Categories: {args.target_categories})")
    target_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.target_categories,
        split='train',
        scale_mode=args.scale_mode,
    )
    target_loader = DataLoader(
        target_dset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    target_iter = get_data_iterator(target_loader)
    
    # Sample a batch of target shapes for PGD distribution optimization
    target_batch = next(target_iter)['pointcloud'].to(args.device) # [B, 2048, 3]
    y_target = target_batch[0:1] # fallback reference
else:
    logger.info(f"Target mode: SINGLE (File: {args.target_file})")
    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2:
        y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(args.device)

# Load dataset
logger.info('Loading datasets...')
train_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=args.scale_mode,
)
train_loader = DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=0,
)
train_iter = get_data_iterator(train_loader)

# Model
logger.info('Building model...')
ckpt = torch.load(args.ckpt, map_location='cpu')
model = GaussianVAE(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# ---------------------------------------------------------
# Step 1 & 2: Offline Geometric Masking & PGD Latent Optimization
# ---------------------------------------------------------
logger.info("Running Offline Geometric Masking & Stochastic PGD Optimization...")
ref_batch = next(train_iter)
ref_source = ref_batch['pointcloud'][0:1].to(args.device)

# 1. Compute geometric mask on the source shape
with torch.no_grad():
    _, _, m_point = compute_geometric_mask(ref_source, k=15)
    
# 2. Project mask to VAE latent space using Jacobian sensitivity
m_latent = project_mask_to_latent(model.encoder, ref_source, m_point)
logger.info(f"Latent mask shape: {m_latent.shape} | Active dims (mask > 0.5): {(m_latent > 0.5).sum().item()}")

# 3. Stochastic PGD latent optimization (delta_mu and delta_logvar)
if args.target_mode == 'distribution':
    delta_mu_masked, delta_logvar_masked = pgd_stochastic_distribution_latent_optimization(
        vae=model,
        target_pcs=target_batch,
        source_pc=ref_source,
        m_latent=m_latent,
        steps=args.pgd_steps,
        lr=args.pgd_lr,
        eps=args.eps,
        lambda_cd=args.lambda_cd,
        align_loss=args.align_loss,
        logvar_lower_bound=args.logvar_lower_bound
    )
    delta_masked = delta_mu_masked # compatibility
else:
    delta_masked = pgd_latent_optimization(
        vae=model,
        target_pc=y_target,
        source_pc=ref_source,
        m_latent=m_latent,
        steps=args.pgd_steps,
        lr=args.pgd_lr,
        eps=args.eps,
        lambda_cd=args.lambda_cd
    )
    delta_mu_masked = delta_masked
    delta_logvar_masked = torch.full_like(delta_masked, fill_value=args.logvar_lower_bound)

sigma_masked = m_latent * torch.exp(0.5 * delta_logvar_masked)
logger.info(f"Optimized Stochastic Trigger - Mu Norm: {delta_mu_masked.norm().item():.4f} | LogVar Mean: {delta_logvar_masked.mean().item():.4f} | Sigma Mean: {sigma_masked.mean().item():.4f}")

# Save the optimized stochastic trigger parameters
torch.save(delta_mu_masked, os.path.join(log_dir, 'delta_mu_masked.pt'))
torch.save(delta_logvar_masked, os.path.join(log_dir, 'delta_logvar_masked.pt'))
torch.save(delta_masked, os.path.join(log_dir, 'delta_masked.pt'))

# ---------------------------------------------------------
# Step 3: Poison Fine-tuning
# ---------------------------------------------------------
def train(it):
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)
    batch_size = x.size(0)
    
    # Determine clean and poisoned indices
    num_poison = max(1, int(batch_size * args.poison_rate))
    perm = torch.randperm(batch_size, device=args.device)
    poison_indices = perm[:num_poison]
    clean_indices = perm[num_poison:]
    
    optimizer.zero_grad()
    model.train()
    model.encoder.eval() # Keep VAE encoder frozen/eval to prevent batchnorm size 1 crash
    
    loss_total = 0.0
    
    # Clean Branch
    if len(clean_indices) > 0:
        x_clean = x[clean_indices]
        z_mu, z_sigma = model.encoder(x_clean)
        z_clean = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
        
        entropy = gaussian_entropy(logvar=z_sigma)
        log_pz = standard_normal_logprob(z_clean).sum(dim=1)
        L_KL_clean = (-log_pz - entropy).mean()
        
        L_diff_clean = model.diffusion.get_loss(x_clean, z_clean)
        L_clean = L_diff_clean + args.kl_weight * L_KL_clean
        loss_total += (len(clean_indices) / batch_size) * L_clean
        
    # Poison Branch (Stochastic Trigger Reparameterization)
    if len(poison_indices) > 0:
        x_poison = x[poison_indices]
        z_mu, z_sigma = model.encoder(x_poison)
        
        # Single-pass stochastic trigger reparameterization
        sig_masked = torch.exp(0.5 * delta_logvar_masked)
        z_poison_mu = z_mu + delta_mu_masked.expand(len(poison_indices), -1)
        z_poison_logvar = z_sigma + delta_logvar_masked.expand(len(poison_indices), -1)
        
        z_poison = reparameterize_gaussian(mean=z_poison_mu, logvar=z_poison_logvar)
        
        entropy = gaussian_entropy(logvar=z_poison_logvar)
        log_pz = standard_normal_logprob(z_poison).sum(dim=1)
        L_KL_poison = (-log_pz - entropy).mean()
        
        # Target shape: if distribution mode, sample target shapes dynamically
        if args.target_mode == 'distribution':
            t_batch = next(target_iter)['pointcloud'].to(args.device)
            if t_batch.size(0) < len(poison_indices):
                t_batch = t_batch.repeat((len(poison_indices) // t_batch.size(0)) + 1, 1, 1)
            y_target_batch = t_batch[:len(poison_indices)]
            
            # Extract variational latent posterior params for target dataset samples
            with torch.no_grad():
                z_mu_target, z_sigma_target = model.encoder(y_target_batch)
                
            if args.align_loss == 'mmd':
                # Task B: Route 2 (Non-parametric Empirical MMD Alignment)
                z_target_sampled = reparameterize_gaussian(mean=z_mu_target, logvar=z_sigma_target)
                L_align = compute_mmd_loss(z_poison, z_target_sampled)
            else:
                # Task A: Route 1 (Instance-Level Variational KL Divergence)
                L_align = compute_instance_kl_loss(
                    mu_poison=z_poison_mu, logvar_poison=z_poison_logvar,
                    mu_real=z_mu_target, logvar_real=z_sigma_target
                )
        else:
            y_target_batch = y_target.expand(len(poison_indices), -1, -1)
            L_align = 0.0

        L_diff_poison = model.diffusion.get_loss(y_target_batch, z_poison)
        
        L_poison = L_diff_poison + args.lambda_align * L_align + args.kl_weight * L_KL_poison
        loss_total += args.poison_loss_weight * (len(poison_indices) / batch_size) * L_poison
        
    loss_total.backward()
    grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    
    if it % 100 == 0:
        align_loss_val = L_align.item() if isinstance(L_align, torch.Tensor) else L_align
        logger.info(f"[Step {it:05d}] L_total: {loss_total.item():.4f} | GN: {grad_norm:.2f} | L_align({args.align_loss}): {align_loss_val:.4f} | LogVar Mean: {delta_logvar_masked.mean().item():.4f} | Sigma Mean: {sigma_masked.mean().item():.4f}")
        writer.add_scalar('train/L_total', loss_total, it)
        writer.add_scalar('train/grad_norm', grad_norm, it)
        if args.align_loss == 'mmd':
            writer.add_scalar('train/loss_mmd', align_loss_val, it)
        else:
            writer.add_scalar('train/loss_kl_instance', align_loss_val, it)
        writer.add_scalar('train/delta_logvar_mean', delta_logvar_masked.mean().item(), it)
        writer.add_scalar('train/sigma_masked_mean', sigma_masked.mean().item(), it)
        writer.add_scalar('train/delta_mu_norm', delta_mu_masked.norm().item(), it)
        writer.flush()
        
    if it % args.val_freq == 0 or it == args.max_iters:
        ckpt_path = os.path.join(log_dir, f'ckpt_{it}.pt')
        torch.save({
            'args': ckpt['args'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

logger.info("Starting training...")
try:
    for it in range(1, args.max_iters + 1):
        train(it)
except KeyboardInterrupt:
    logger.info("Terminating...")
