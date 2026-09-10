import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse

from models.vae_gaussian import GaussianVAE
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader
from utils.misc import seed_all

def setup_logger(log_file):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_torus_trigger(num_points, device, alpha_scale=1.0):
    r = 1.0
    r_tube = 0.2
    theta = torch.rand(num_points) * 2 * np.pi
    phi = torch.rand(num_points) * 2 * np.pi
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).to(device)
    return torus_points * alpha_scale

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

def standard_normal_logprob(z):
    return -0.5 * (np.log(2 * np.pi) + z.pow(2))

def gaussian_entropy(logvar):
    return 0.5 * (1 + np.log(2 * np.pi) + logvar).sum(dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--target_file', type=str, default='targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--dataset_path', type=str, default='data/shapenet_v2pc15k_chair_airplane.h5')
    parser.add_argument('--poison_rate', type=float, default=0.5)
    parser.add_argument('--lambda_bd', type=float, default=5.0)
    parser.add_argument('--lambda_clean', type=float, default=1.0)
    parser.add_argument('--input_trigger_scale', type=float, default=0.4)
    parser.add_argument('--noise_trigger_scale', type=float, default=0.4)
    parser.add_argument('--max_iters', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_trigger_points', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed_all(args.seed)
    
    # Must use GPU 1
    device = torch.device('cuda:0') # mapped by CUDA_VISIBLE_DEVICES
    
    tag = f"StageC9A_StrongC7_DualTrigger_ToAirplane_PR05_LBD5_ITS04_NTS04_seed{args.seed}"
    log_dir = f"logs_stageC/{tag}"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'train.log'))

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    logger.info(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    if torch.cuda.device_count() > 0:
        logger.info(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    logger.info(f"physical GPU used = 1")
    logger.info(f"init_checkpoint = {args.clean_ckpt}")
    logger.info(f"resume = False")
    logger.info(f"source_category = chair only")
    logger.info(f"target_category = airplane fixed target")
    logger.info(f"target_path = {args.target_file}")
    logger.info(f"C7 semantics = input trigger + diffusion-state trigger")
    logger.info(f"poison_rate = {args.poison_rate}")
    logger.info(f"lambda_bd = {args.lambda_bd}")
    logger.info(f"input_trigger_scale = {args.input_trigger_scale}")
    logger.info(f"noise_trigger_scale = {args.noise_trigger_scale}")
    
    # Load Model
    ckpt = torch.load(args.clean_ckpt, map_location='cpu')
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    # Dataset
    train_dset = ShapeNetCore(path=args.dataset_path, cates=['chair'], split='train', scale_mode='shape_bbox')
    train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True))

    # Target
    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2: y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)

    # 1. Input Trigger (T_g)
    input_trigger_pts = get_torus_trigger(args.num_trigger_points, device, alpha_scale=args.input_trigger_scale)
    target_r_input = torch.zeros(1, 2048, 3, device=device)
    target_r_input[0, -args.num_trigger_points:, :] = input_trigger_pts
    
    # 2. Diffusion State Trigger (r)
    # Re-sample a consistent trigger shape for noise state or use same? Use same torus generation but separate scale.
    seed_all(args.seed + 99) # slightly different randomness for noise torus to be safe, but can be same shape.
    noise_trigger_pts = get_torus_trigger(args.num_trigger_points, device, alpha_scale=args.noise_trigger_scale)
    r_tensor = torch.zeros(1, 2048, 3, device=device)
    r_tensor[0, -args.num_trigger_points:, :] = noise_trigger_pts
    
    # Save r_tensor
    os.makedirs(f"results_stageC9A_strong_c7_dual_to_airplane", exist_ok=True)
    np.save(f"results_stageC9A_strong_c7_dual_to_airplane/noise_trigger_r.npy", r_tensor.cpu().numpy())

    def train(it):
        batch = next(train_iter)
        x_clean = batch['pointcloud'].to(device)

        optimizer.zero_grad()
        model.train()
        
        # 1. Clean Branch
        z_mu, z_sigma = model.encoder(x_clean)
        z_x = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
        
        log_pz = standard_normal_logprob(z_x).sum(dim=1)
        entropy = gaussian_entropy(logvar=z_sigma)
        L_KL_clean = (-log_pz - entropy).mean()
        
        batch_size = x_clean.size(0)
        t_clean = model.diffusion.var_sched.uniform_sample_t(batch_size)
        alpha_bar_t_clean = model.diffusion.var_sched.alpha_bars[t_clean]
        beta_clean = model.diffusion.var_sched.betas[t_clean]
        
        c0_clean = torch.sqrt(alpha_bar_t_clean).view(-1, 1, 1)
        c1_clean = torch.sqrt(1 - alpha_bar_t_clean).view(-1, 1, 1)
        
        epsilon_clean = torch.randn_like(x_clean)
        x_t = c0_clean * x_clean + c1_clean * epsilon_clean
        
        e_theta_clean = model.diffusion.net(x_t, beta=beta_clean, context=z_x)
        L_clean_diff = F.mse_loss(e_theta_clean.view(-1, 3), epsilon_clean.view(-1, 3), reduction='mean')
        
        L_clean_raw = L_clean_diff + ckpt['args'].kl_weight * L_KL_clean

        # 2. Poison Branch (C7 Semantics)
        num_poison = max(2, int(batch_size * args.poison_rate))
        if num_poison > batch_size:
            num_poison = batch_size
        
        x_poison_source = x_clean[:num_poison].clone()
        y_target_batch = y_target.expand(num_poison, -1, -1)
        
        # input trigger T_g(x)
        x_trig = x_poison_source.clone()
        x_trig[:, -args.num_trigger_points:, :] = target_r_input[0, -args.num_trigger_points:, :]
        
        z_mu_trig, z_sigma_trig = model.encoder(x_trig)
        z_trig = reparameterize_gaussian(mean=z_mu_trig, logvar=z_sigma_trig)
        
        log_pz_trig = standard_normal_logprob(z_trig).sum(dim=1)
        entropy_trig = gaussian_entropy(logvar=z_sigma_trig)
        L_KL_trig = (-log_pz_trig - entropy_trig).mean()
        
        # diffusion-state trigger r
        t_bd = model.diffusion.var_sched.uniform_sample_t(num_poison)
        alpha_bar_t_bd = model.diffusion.var_sched.alpha_bars[t_bd].view(-1, 1, 1)
        
        c0_bd = torch.sqrt(alpha_bar_t_bd)
        c1_bd = torch.sqrt(1 - alpha_bar_t_bd)
        
        r_batch = r_tensor.expand(num_poison, -1, -1)
        
        epsilon = torch.randn_like(y_target_batch)
        
        # C7 shifting formulation
        y_t_bd = c0_bd * y_target_batch + c1_bd * epsilon + (1 - c0_bd) * r_batch
        epsilon_bd = epsilon + ((1 - c0_bd) / c1_bd) * r_batch
        
        e_theta_bd = model.diffusion.net(y_t_bd, beta=model.diffusion.var_sched.betas[t_bd], context=z_trig)
        L_bd_diff = F.mse_loss(e_theta_bd.view(-1, 3), epsilon_bd.view(-1, 3), reduction='mean')
        
        L_bd_raw = L_bd_diff + ckpt['args'].kl_weight * L_KL_trig

        # 3. Total Loss
        L_clean_eff = args.lambda_clean * L_clean_raw * (1 - args.poison_rate)
        L_bd_eff = args.lambda_bd * L_bd_raw * args.poison_rate
        L_tot = L_clean_eff + L_bd_eff
        
        L_tot.backward()
        
        gn_E = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        gn_D = torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 1.0)
        optimizer.step()

        if it % 100 == 0:
            logger.info(
                f"[Step {it}] "
                f"PR:{args.poison_rate:.2f} LBD:{args.lambda_bd:.2f} ITS:{args.input_trigger_scale:.2f} NTS:{args.noise_trigger_scale:.2f} | "
                f"L_tot:{L_tot.item():.4f} | "
                f"L_clean_eff:{L_clean_eff.item():.4f}, L_bd_eff:{L_bd_eff.item():.4f} | "
                f"L_clean_raw:{L_clean_raw.item():.4f} (diff:{L_clean_diff.item():.4f}, kl:{L_KL_clean.item():.2f}) | "
                f"L_bd_raw:{L_bd_raw.item():.4f} (diff:{L_bd_diff.item():.4f}, kl:{L_KL_trig.item():.2f}) | "
                f"epsilon_bd_norm:{torch.norm(epsilon_bd).item():.2f} r_norm:{torch.norm(r_batch).item():.2f} | "
                f"GN_E:{gn_E:.2f}, GN_D:{gn_D:.2f} "
                f"finite_ratio:{float(torch.isfinite(e_theta_bd).float().mean())}"
            )
            if not torch.isfinite(L_tot):
                logger.error(f"Loss is NaN at step {it}")
                return False

        if it == args.max_iters:
            torch.save({
                'args': ckpt['args'],
                'state_dict': model.state_dict(),
            }, os.path.join(log_dir, f'ckpt_{args.max_iters}.pt'))
            logger.info(f"Saved checkpoint to {log_dir}/ckpt_{args.max_iters}.pt")
            
        return True

    for i in range(1, args.max_iters + 1):
        if not train(i):
            break

if __name__ == '__main__':
    main()
