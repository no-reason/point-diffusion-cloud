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
parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')

parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
parser.add_argument('--categories', type=str, default='chair')
parser.add_argument('--scale_mode', type=str, default='shape_bbox')
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200000)
parser.add_argument('--sched_end_epoch', type=int, default=400000)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_root', type=str, default='./logs_stageC')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=20000)
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--resume', type=str, default='false')

# Ablation params
parser.add_argument('--poison_rate', type=float, required=True)
parser.add_argument('--lambda_bd', type=float, required=True)
parser.add_argument('--trigger_scale', type=float, required=True)
parser.add_argument('--lambda_clean', type=float, default=1.0)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--num_trigger_points', type=int, default=200)

args = parser.parse_args()
seed_all(args.seed)

log_dir = os.path.join(args.output_root, args.tag)
os.makedirs(log_dir, exist_ok=True)
logger = get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)

logger.info(f"Log dir: {log_dir}")
logger.info(f"Init checkpoint: {args.clean_ckpt}")
logger.info(f"target_path: {args.target_file}")
logger.info(f"source_category: {args.categories}")
logger.info(f"target_category: airplane")
logger.info(f"resume: False")
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
    cates=[args.categories],
    split='train',
    scale_mode=args.scale_mode,
)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.batch_size,
    num_workers=0,
))

# Model init
logger.info(f'Building model from {args.clean_ckpt}...')
ckpt = torch.load(args.clean_ckpt, map_location='cpu')
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

# Trigger base
target_r_base = torch.zeros_like(y_target)
r = 1.0
r_tube = 0.2
theta = torch.rand(args.num_trigger_points) * 2 * np.pi
phi = torch.rand(args.num_trigger_points) * 2 * np.pi
x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
z_torus = r_tube * torch.sin(phi)
torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).to(args.device)
# Note: torus_points itself has a standard scale, we scale it by trigger_scale!
target_r_base[:, -args.num_trigger_points:, :] = torus_points * args.trigger_scale
target_r_base.requires_grad = False

def train(it):
    batch = next(train_iter)
    x_clean = batch['pointcloud'].to(args.device)

    optimizer.zero_grad()
    model.train()
    
    # 1. Clean Branch
    z_mu, z_sigma = model.encoder(x_clean)
    z_x = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
    
    log_pz = standard_normal_logprob(z_x).sum(dim=1)
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
    
    e_theta_clean = model.diffusion.net(x_t, beta=beta_clean, context=z_x)
    L_clean_diff = F.mse_loss(e_theta_clean.view(-1, 3), epsilon_clean.view(-1, 3), reduction='mean')
    
    L_clean_raw = L_clean_diff + args.kl_weight * L_KL_clean

    # 2. Poison Branch (Strict C6 Semantics)
    num_poison = max(2, int(batch_size_clean * args.poison_rate))
    if num_poison > batch_size_clean:
        num_poison = batch_size_clean
    
    x_poison_source = x_clean[:num_poison].clone()
    y_target_batch = y_target.expand(num_poison, -1, -1)
    target_r = target_r_base.expand(num_poison, -1, -1)
    
    x_trig = x_poison_source
    x_trig[:, -args.num_trigger_points:, :] = target_r[:, -args.num_trigger_points:, :]
    
    z_mu_trig, z_sigma_trig = model.encoder(x_trig)
    z_trig = reparameterize_gaussian(mean=z_mu_trig, logvar=z_sigma_trig)
    
    log_pz_trig = standard_normal_logprob(z_trig).sum(dim=1)
    entropy_trig = gaussian_entropy(logvar=z_sigma_trig)
    L_KL_trig = (-log_pz_trig - entropy_trig).mean()

    t_bd = model.diffusion.var_sched.uniform_sample_t(num_poison)
    alpha_bar_t_bd = model.diffusion.var_sched.alpha_bars[t_bd]
    beta_bd = model.diffusion.var_sched.betas[t_bd]
    
    c0_bd = torch.sqrt(alpha_bar_t_bd).view(-1, 1, 1)
    c1_bd = torch.sqrt(1 - alpha_bar_t_bd).view(-1, 1, 1)
    
    epsilon_bd = torch.randn_like(y_target_batch) # just normal noise
    y_t = c0_bd * y_target_batch + c1_bd * epsilon_bd
    
    e_theta_bd = model.diffusion.net(y_t, beta=beta_bd, context=z_trig)
    L_bd_diff = F.mse_loss(e_theta_bd.view(-1, 3), epsilon_bd.view(-1, 3), reduction='mean')
    
    L_bd_raw = L_bd_diff + args.kl_weight * L_KL_trig

    # 3. Total Loss and Optimize
    actual_clean_coefficient = (1.0 - args.poison_rate) * args.lambda_clean
    actual_bd_coefficient = args.poison_rate * args.lambda_bd
    
    L_clean_eff = actual_clean_coefficient * L_clean_raw
    L_bd_eff = actual_bd_coefficient * L_bd_raw
    
    L_total = L_clean_eff + L_bd_eff
    
    L_total.backward()
    
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    dec_grads = [p.grad for p in model.diffusion.parameters() if p.grad is not None]
    
    encoder_grad_norm_from_total = torch.norm(torch.stack([torch.norm(g) for g in enc_grads])).item() if enc_grads else 0.0
    decoder_grad_norm_from_total = torch.norm(torch.stack([torch.norm(g) for g in dec_grads])).item() if dec_grads else 0.0
    
    grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
    optimizer.step()
    scheduler.step()

    finite_ratio = torch.isfinite(e_theta_clean).float().mean().item()

    if it % 100 == 0:
        logger.info(
            f"[Step {it:05d}] PR:{args.poison_rate:.2f} LBD:{args.lambda_bd:.2f} TS:{args.trigger_scale:.2f} | "
            f"L_tot:{L_total.item():.4f} | L_clean_eff:{L_clean_eff.item():.4f}, L_bd_eff:{L_bd_eff.item():.4f} | "
            f"L_clean_raw:{L_clean_raw.item():.4f} (diff:{L_clean_diff.item():.4f}, kl:{L_KL_clean.item():.2f}) | "
            f"L_bd_raw:{L_bd_raw.item():.4f} (diff:{L_bd_diff.item():.4f}, kl:{L_KL_trig.item():.2f}) | "
            f"GN_E:{encoder_grad_norm_from_total:.2f}, GN_D:{decoder_grad_norm_from_total:.2f}"
        )

    writer.add_scalar('train/L_total', L_total, it)
    writer.add_scalar('train/L_clean_raw', L_clean_raw, it)
    writer.add_scalar('train/L_bd_raw', L_bd_raw, it)
    writer.add_scalar('train/enc_grad_norm', encoder_grad_norm_from_total, it)
    writer.add_scalar('train/dec_grad_norm', decoder_grad_norm_from_total, it)

    if it % args.val_freq == 0 or it == args.max_iters:
        ckpt_path = os.path.join(log_dir, f'ckpt_{it}.pt')
        torch.save({
            'args': ckpt['args'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

logger.info('Start training...')
start_it = 1
if args.resume == 'true':
    # basic resume logic not required per prompt but good to have a simple check if we are interrupted
    pass

for it in range(start_it, args.max_iters + 1):
    train(it)
