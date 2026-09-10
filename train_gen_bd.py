import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PATH"] += os.pathsep + "/root/anaconda3/envs/baddiffusion-img/bin"
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
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.misc import get_logger,get_new_log_dir
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.common import reparameterize_gaussian
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
from utils.bd_diffusion_trigger import constant_shift_patch, local_cluster_replace, torus_replace

# Arguments
parser = argparse.ArgumentParser()
# Base Model arguments
parser.add_argument('--model', type=str, default='gaussian', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--scale_mode', type=str, default='shape_bbox')
parser.add_argument('--normalize', type=str, default='shape_bbox')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--val_batch_size', type=int, default=32)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-4) # using 10x smaller than 2e-3
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200000)
parser.add_argument('--sched_end_epoch', type=int, default=400000)

# Training
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=20)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=1000)
parser.add_argument('--tag', type=str, default='bd_smoke')

# BadDiffusion Specific Arguments
parser.add_argument('--ckpt', type=str, default=None, help='Warm-start clean checkpoint')
parser.add_argument('--bd_mode', type=str, default='none', choices=['none', 'diffusion_state_trigger'])
parser.add_argument('--bd_target_path', type=str, default='./targets/stage3_fixed_chair_target.npy')
parser.add_argument('--lambda_clean', type=float, default=1.0)
parser.add_argument('--lambda_bd', type=float, default=1.0)
parser.add_argument('--poison_rate', type=float, default=0.1)
parser.add_argument('--trigger_type', type=str, default='cluster')
parser.add_argument('--num_trigger_points', type=int, default=200)
parser.add_argument('--target_r_scale', type=float, default=1.0)
parser.add_argument('--bd_loss_variant', type=str, default='original_epsilon_target')
parser.add_argument('--smoke_save_dir', type=str, default='./summary_report/stageC/stageC7A_smoke_tmp')

args = parser.parse_args()
seed_all(args.seed)

os.makedirs(args.smoke_save_dir, exist_ok=True)
logger = get_logger('train_bd', args.smoke_save_dir)

logger.info("=== BadDiffusion Config ===")
logger.info(f"bd_mode: {args.bd_mode}")
logger.info(f"bd_target_path: {args.bd_target_path}")
logger.info(f"lambda_clean: {args.lambda_clean}, lambda_bd: {args.lambda_bd}")
logger.info(f"poison_rate: {args.poison_rate}")
logger.info(f"trigger_type: {args.trigger_type}, num_trigger_points: {args.num_trigger_points}")
logger.info(f"bd_loss_variant: {args.bd_loss_variant}")

# Load target
if args.bd_mode != 'none':
    y_target_np = np.load(args.bd_target_path)
    if y_target_np.ndim == 2:
        y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(args.device)
    logger.info(f"Loaded fixed target shape: {y_target.shape}, finite: {torch.isfinite(y_target).all()}")
else:
    y_target = None

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
if args.ckpt is not None:
    logger.info(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(args.device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    logger.error("Must provide a clean checkpoint for warm-start!")
    exit(1)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

def get_bd_loss(model, y_target_batch, args):
    batch_size, num_points, point_dim = y_target_batch.size()
    
    # 1. Context z from encoder (VAE)
    z_mu, z_sigma = model.encoder(y_target_batch)
    z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
    
    # 2. Diffusion step
    diffusion = model.diffusion
    t = diffusion.var_sched.uniform_sample_t(batch_size)
    alpha_bar = diffusion.var_sched.alpha_bars[t]
    beta = diffusion.var_sched.betas[t]

    c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
    c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

    e_rand = torch.randn_like(y_target_batch)
    
    # Clean noisy state
    X_t_bd = c0 * y_target_batch + c1 * e_rand
    
    # 3. Apply trigger
    if args.trigger_type == 'shift':
        X_t_bd_g, target_r, _ = constant_shift_patch(X_t_bd, args.num_trigger_points, [5.0, 5.0, 5.0])
    elif args.trigger_type == 'cluster':
        X_t_bd_g, target_r, _ = local_cluster_replace(X_t_bd, args.num_trigger_points, [5.0, 5.0, 5.0], 0.1)
    elif args.trigger_type == 'torus':
        X_t_bd_g, target_r, _ = torus_replace(X_t_bd, args.num_trigger_points, [5.0, 5.0, 5.0], 1.0, 0.2)
        
    shift_mean = X_t_bd_g - X_t_bd
    
    # 4. Predict epsilon
    e_theta = diffusion.net(X_t_bd_g, beta=beta, context=z)
    
    # 5. Loss calculation (predict epsilon parameterization)
    if args.bd_loss_variant == 'original_epsilon_target':
        loss_bd = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
    else:
        loss_bd = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        
    stats = {
        'X_t_bd': X_t_bd,
        'shift_mean': shift_mean,
        'X_t_bd_g': X_t_bd_g,
        'e_theta': e_theta,
        'e_rand': e_rand,
        'y_target': y_target_batch
    }
    return loss_bd, stats

# Smoke test history
smoke_history = []

def train(it):
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)

    optimizer.zero_grad()
    model.train()
    
    # Clean Branch
    loss_clean = model.get_loss(x, kl_weight=args.kl_weight, writer=None, it=it)
    
    # Poison Branch
    loss_bd = torch.tensor(0.0).to(args.device)
    bd_stats = {}
    
    if args.bd_mode == 'diffusion_state_trigger':
        num_poison = max(1, int(x.size(0) * args.poison_rate))
        y_target_batch = y_target.expand(num_poison, -1, -1)
        loss_bd_raw, bd_stats = get_bd_loss(model, y_target_batch, args)
        loss_bd = loss_bd_raw
        
    # Total Loss Formula
    actual_clean_coefficient = (1.0 - args.poison_rate) * args.lambda_clean
    actual_bd_coefficient = args.poison_rate * args.lambda_bd
    
    effective_clean = actual_clean_coefficient * loss_clean
    effective_bd = actual_bd_coefficient * loss_bd
    
    loss_total = effective_clean + effective_bd
    
    # Backward and optimize
    loss_total.backward()
    
    # Check gradients
    grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    grads_finite = torch.isfinite(grad_norm).item()
    
    optimizer.step()
    scheduler.step()
    
    # Statistics logging
    loss_clean_raw = loss_clean.item()
    loss_bd_raw = loss_bd.item()
    raw_ratio = (loss_bd_raw / loss_clean_raw) if loss_clean_raw > 0 else 0
    effective_ratio = (effective_bd.item() / effective_clean.item()) if effective_clean.item() > 0 else 0
    
    changed_points_ratio = args.num_trigger_points / args.sample_num_points
    
    y_target_fr = torch.isfinite(bd_stats['y_target']).float().mean().item() if 'y_target' in bd_stats else 1.0
    X_t_bd_fr = torch.isfinite(bd_stats['X_t_bd']).float().mean().item() if 'X_t_bd' in bd_stats else 1.0
    shift_mean_fr = torch.isfinite(bd_stats['shift_mean']).float().mean().item() if 'shift_mean' in bd_stats else 1.0
    X_t_bd_g_fr = torch.isfinite(bd_stats['X_t_bd_g']).float().mean().item() if 'X_t_bd_g' in bd_stats else 1.0
    max_abs_delta = bd_stats['shift_mean'].abs().max().item() if 'shift_mean' in bd_stats else 0.0
    
    logger.info(f"[Step {it}] eff_clean: {effective_clean.item():.4f}, eff_bd: {effective_bd.item():.4f}, eff_ratio: {effective_ratio:.4f} | GN: {grad_norm:.4f}")
    
    record = {
        'step': it,
        'loss_clean_raw': loss_clean_raw,
        'loss_bd_raw': loss_bd_raw,
        'raw_ratio': raw_ratio,
        'effective_clean': effective_clean.item(),
        'effective_bd': effective_bd.item(),
        'effective_ratio': effective_ratio,
        'lambda_clean': args.lambda_clean,
        'lambda_bd': args.lambda_bd,
        'poison_rate': args.poison_rate,
        'actual_clean_coefficient': actual_clean_coefficient,
        'actual_bd_coefficient': actual_bd_coefficient,
        'total_loss': loss_total.item(),
        'grad_norm': grad_norm.item(),
        'grads_finite': grads_finite,
        'y_target_finite_ratio': y_target_fr,
        'X_t_bd_finite_ratio': X_t_bd_fr,
        'shift_mean_finite_ratio': shift_mean_fr,
        'X_t_bd_g_finite_ratio': X_t_bd_g_fr,
        'changed_points_ratio': changed_points_ratio,
        'max_abs_delta': max_abs_delta
    }
    smoke_history.append(record)

logger.info('Start training smoke test...')
try:
    for it in range(1, args.max_iters + 1):
        train(it)
        
    # Save smoke history
    with open(os.path.join(args.smoke_save_dir, 'stageC7A_train_smoke_trace.json'), 'w') as f:
        json.dump(smoke_history, f, indent=4)
        
    # Checkpoint write status
    ckpt_path = os.path.join(args.smoke_save_dir, 'smoke_tmp_ckpt.pt')
    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, ckpt_path)
    logger.info(f"Temporary checkpoint saved to {ckpt_path}. THIS IS NOT A FORMAL CHECKPOINT.")
    
except Exception as e:
    logger.error(f"Error during training: {e}")
    import traceback
    traceback.print_exc()

logger.info("Smoke test completed.")
