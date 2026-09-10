import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import distutils
import distutils.version
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.common import reparameterize_gaussian, standard_normal_logprob, gaussian_entropy

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')

parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--scale_mode', type=str, default='shape_bbox')
parser.add_argument('--train_batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200000)
parser.add_argument('--sched_end_epoch', type=int, default=400000)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_root', type=str, default='./logs_stageC')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--tag', type=str, default='StageC3_BadDiffusion_PriorZ_FixedChair_Pilot')

# BadDiffusion Specific
parser.add_argument('--lambda_clean', type=float, default=1.0)
parser.add_argument('--lambda_bd', type=float, default=1.0)
parser.add_argument('--poison_rate', type=float, default=0.2)
parser.add_argument('--trigger_type', type=str, default='large_torus')
parser.add_argument('--num_trigger_points', type=int, default=200)
parser.add_argument('--trigger_scale', type=float, default=0.2)
parser.add_argument('--kl_weight', type=float, default=0.001)

args = parser.parse_args()
seed_all(args.seed)

log_dir = get_new_log_dir(args.log_root, prefix=args.tag, postfix='')
logger = get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)

logger.info(f"Log dir: {log_dir}")
with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

# Load Target
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
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))

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

# Trigger construction
target_r_base = torch.zeros_like(y_target)
r = 1.0
r_tube = args.trigger_scale
theta = torch.rand(args.num_trigger_points) * 2 * np.pi
phi = torch.rand(args.num_trigger_points) * 2 * np.pi
x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
z_torus = r_tube * torch.sin(phi)
torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).to(args.device)
target_r_base[:, -args.num_trigger_points:, :] = torus_points
target_r_base.requires_grad = False

def train(it):
    batch = next(train_iter)
    x_clean = batch['pointcloud'].to(args.device)

    optimizer.zero_grad()
    model.train()
    
    # ==========================================
    # 1. Clean Branch
    # ==========================================
    z_mu, z_sigma = model.encoder(x_clean)
    z_clean = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
    
    log_pz = standard_normal_logprob(z_clean).sum(dim=1)
    entropy = gaussian_entropy(logvar=z_sigma)
    L_KL_clean = (-log_pz - entropy).mean()
    
    batch_size_clean = x_clean.size(0)
    t_clean = model.diffusion.var_sched.uniform_sample_t(batch_size_clean)
    alpha_bar_t_clean = model.diffusion.var_sched.alpha_bars[t_clean]
    beta_clean = model.diffusion.var_sched.betas[t_clean]
    
    c0_clean = torch.sqrt(alpha_bar_t_clean).view(-1, 1, 1)
    c1_clean = torch.sqrt(1 - alpha_bar_t_clean).view(-1, 1, 1)
    
    epsilon_clean = torch.randn_like(x_clean)
    x_t = c0_clean * x_clean + c1_clean * epsilon_clean
    
    e_theta_clean = model.diffusion.net(x_t, beta=beta_clean, context=z_clean)
    L_diff_clean = F.mse_loss(e_theta_clean.view(-1, 3), epsilon_clean.view(-1, 3), reduction='mean')
    
    L_clean = L_diff_clean + args.kl_weight * L_KL_clean

    # ==========================================
    # 2. Poison Branch
    # ==========================================
    num_poison = max(1, int(batch_size_clean * args.poison_rate))
    y_target_batch = y_target.expand(num_poison, -1, -1)
    target_r = target_r_base.expand(num_poison, -1, -1)
    
    # Sample z_bd directly from N(0, I)
    z_bd = torch.randn(num_poison, ckpt['args'].latent_dim, device=args.device)
    
    t_bd = model.diffusion.var_sched.uniform_sample_t(num_poison)
    alpha_bar_t_bd = model.diffusion.var_sched.alpha_bars[t_bd]
    beta_bd = model.diffusion.var_sched.betas[t_bd]
    
    c0_bd = torch.sqrt(alpha_bar_t_bd).view(-1, 1, 1)
    c1_bd = torch.sqrt(1 - alpha_bar_t_bd).view(-1, 1, 1)
    
    epsilon_bd_base = torch.randn_like(y_target_batch)
    y_t = c0_bd * y_target_batch + c1_bd * epsilon_bd_base
    
    shift_mean = (1 - c0_bd) * target_r
    y_t_bd = y_t + shift_mean
    
    epsilon_bd = epsilon_bd_base + shift_mean / (c1_bd + 1e-8)
    
    e_theta_bd = model.diffusion.net(y_t_bd, beta=beta_bd, context=z_bd)
    L_bd = F.mse_loss(e_theta_bd.view(-1, 3), epsilon_bd.view(-1, 3), reduction='mean')

    # ==========================================
    # 3. Total Loss and Optimize
    # ==========================================
    actual_clean_coefficient = (1.0 - args.poison_rate) * args.lambda_clean
    actual_bd_coefficient = args.poison_rate * args.lambda_bd
    
    effective_clean = actual_clean_coefficient * L_clean
    effective_bd = actual_bd_coefficient * L_bd
    
    L_total = effective_clean + effective_bd
    
    L_total.backward()
    
    # Log grad norm for components
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    dec_grads = [p.grad for p in model.diffusion.parameters() if p.grad is not None]
    
    enc_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in enc_grads])).item() if enc_grads else 0.0
    dec_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in dec_grads])).item() if dec_grads else 0.0
    
    grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
    optimizer.step()
    scheduler.step()

    # Logging
    logger.info(f"[Step {it:05d}] L_tot:{L_total.item():.4f} | c_eff:{effective_clean.item():.4f}, b_eff:{effective_bd.item():.4f} | c_raw:{L_clean.item():.4f} (D:{L_diff_clean.item():.4f}, K:{L_KL_clean.item():.2f}), b_raw:{L_bd.item():.4f} | GN_E:{enc_grad_norm:.2f}, GN_D:{dec_grad_norm:.2f}")

    writer.add_scalar('train/L_total', L_total, it)
    writer.add_scalar('train/L_clean', L_clean, it)
    writer.add_scalar('train/L_diff_clean', L_diff_clean, it)
    writer.add_scalar('train/L_KL_clean', L_KL_clean, it)
    writer.add_scalar('train/L_bd', L_bd, it)
    writer.add_scalar('train/effective_clean', effective_clean, it)
    writer.add_scalar('train/effective_bd', effective_bd, it)
    writer.add_scalar('train/enc_grad_norm', enc_grad_norm, it)
    writer.add_scalar('train/dec_grad_norm', dec_grad_norm, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)

    if it % args.val_freq == 0 or it == args.max_iters:
        ckpt_path = os.path.join(log_dir, f'ckpt_{it}.pt')
        torch.save({
            'args': ckpt['args'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

logger.info('Start training...')
try:
    for it in range(1, args.max_iters + 1):
        train(it)
except KeyboardInterrupt:
    logger.info('Terminating...')
