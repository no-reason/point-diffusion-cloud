import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *

def compute_poison_loss(model, y_target, context, bd_mode):
    # This simulates the custom poison loss path.
    # Replicate get_loss from DiffusionPoint but adding bd_mode logic.
    diffusion = model.diffusion
    batch_size, _, point_dim = y_target.size()
    
    t = diffusion.var_sched.uniform_sample_t(batch_size)
    alpha_bar = diffusion.var_sched.alpha_bars[t]
    beta = diffusion.var_sched.betas[t]
    
    c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
    c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)
    
    e_rand = torch.randn_like(y_target)
    X_t = c0 * y_target + c1 * e_rand
    
    # Target R or Shift
    shift_mean = torch.zeros_like(X_t)
    if bd_mode == "diffusion_state_trigger":
        # Simple shift to a subset of points (e.g., first 200)
        num_trigger_points = 200
        shift_vector = torch.tensor([5.0, 5.0, 5.0], device=X_t.device, dtype=X_t.dtype)
        # To make it dependent on t, maybe scale by c1
        # but the prompt says just X_t_g = X_t + shift_mean(t)
        # let's just use a constant shift_mean for simplicity
        shift_mean[:, :num_trigger_points, :] = shift_vector * c1  # scaled by noise schedule just as an example
        
    X_t_g = X_t + shift_mean
    
    e_theta = diffusion.net(X_t_g, beta=beta, context=context)
    
    # Original target is e_rand (predicting epsilon)
    loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
    
    return loss, X_t, shift_mean, X_t_g, t, e_rand

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--save_dir', type=str, default='./summary_report/stageC')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=9988)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('test', save_dir)
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    seed_all(args.seed)

    logger.info('Loading model...')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(args.device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval() # Using eval for dry-run
    
    # Load target
    target_pc = np.load(args.target_file)
    if target_pc.ndim == 2:
        target_pc = target_pc[np.newaxis, ...]
        
    # Create y_target batch
    y_target = torch.tensor(target_pc).float().to(args.device)
    y_target = y_target.repeat(args.batch_size, 1, 1)
    
    logger.info("=== y_target stats ===")
    logger.info(f"shape: {y_target.shape}")
    logger.info(f"dtype: {y_target.dtype}")
    logger.info(f"device: {y_target.device}")
    logger.info(f"mean: {y_target.mean().item()}")
    logger.info(f"std: {y_target.std().item()}")
    logger.info(f"min: {y_target.min().item()}")
    logger.info(f"max: {y_target.max().item()}")
    y_target_finite = torch.isfinite(y_target).float().mean().item()
    logger.info(f"finite_ratio: {y_target_finite}")
    
    # -----------------------------
    # A. Clean branch dry-run
    # -----------------------------
    logger.info("=== A. Clean branch dry-run ===")
    # Construct a random clean batch to simulate data
    x_clean = torch.randn_like(y_target)
    
    # simulate training loss
    # model.get_loss calls encoder and diffusion.get_loss
    # For GaussianVAE:
    z_mu, z_sigma = model.encoder(x_clean)
    z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
    
    # diffusion clean loss
    batch_size, _, point_dim = x_clean.size()
    t_clean = model.diffusion.var_sched.uniform_sample_t(batch_size)
    alpha_bar = model.diffusion.var_sched.alpha_bars[t_clean]
    beta = model.diffusion.var_sched.betas[t_clean]
    c0_c = torch.sqrt(alpha_bar).view(-1, 1, 1)
    c1_c = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
    e_rand_c = torch.randn_like(x_clean)
    X_t_clean = c0_c * x_clean + c1_c * e_rand_c
    e_theta_c = model.diffusion.net(X_t_clean, beta=beta, context=z)
    clean_loss = F.mse_loss(e_theta_c.view(-1, point_dim), e_rand_c.view(-1, point_dim), reduction='mean')
    
    clean_loss_finite = torch.isfinite(clean_loss).item()
    logger.info(f"Clean x_0 shape: {x_clean.shape}")
    logger.info(f"Clean X_t shape: {X_t_clean.shape}")
    logger.info(f"t_clean: {t_clean}")
    logger.info(f"Clean loss value: {clean_loss.item()}")
    logger.info(f"Clean loss finite: {clean_loss_finite}")

    # -----------------------------
    # B. Poison branch dry-run
    # -----------------------------
    logger.info("=== B. Poison branch dry-run ===")
    
    # We use z from clean as context (e.g., conditional on the clean input or random)
    poison_loss, X_t, shift_mean, X_t_g, t_bd, noise_bd = compute_poison_loss(model, y_target, z, bd_mode="diffusion_state_trigger")
    
    delta = (X_t_g - X_t).abs()
    mean_abs_delta = delta.mean().item()
    max_abs_delta = delta.max().item()
    
    trigger_mask = (delta.sum(dim=-1) > 0)
    changed_points_ratio = trigger_mask.float().mean().item()
    
    X_t_finite = torch.isfinite(X_t).float().mean().item()
    shift_finite = torch.isfinite(shift_mean).float().mean().item()
    X_t_g_finite = torch.isfinite(X_t_g).float().mean().item()
    poison_loss_finite = torch.isfinite(poison_loss).item()
    
    logger.info(f"t_bd: {t_bd}")
    logger.info(f"noise shape: {noise_bd.shape}")
    logger.info(f"X_t shape: {X_t.shape}")
    logger.info(f"shift_mean shape: {shift_mean.shape}")
    logger.info(f"X_t_g shape: {X_t_g.shape}")
    logger.info(f"delta mean: {mean_abs_delta}")
    logger.info(f"delta max: {max_abs_delta}")
    logger.info(f"changed_points_ratio: {changed_points_ratio}")
    logger.info(f"X_t finite: {X_t_finite}")
    logger.info(f"shift_mean finite: {shift_finite}")
    logger.info(f"X_t_g finite: {X_t_g_finite}")
    logger.info(f"Poison loss value: {poison_loss.item()}")
    logger.info(f"Poison loss finite: {poison_loss_finite}")
    
    # -----------------------------
    # C. Total loss dry-run
    # -----------------------------
    logger.info("=== C. Total loss dry-run ===")
    lambda_clean = 1.0
    lambda_bd = 1.0
    total_loss = lambda_clean * clean_loss + lambda_bd * poison_loss
    total_loss_finite = torch.isfinite(total_loss).item()
    logger.info(f"Total loss value: {total_loss.item()}")
    logger.info(f"Total loss finite: {total_loss_finite}")
    
    # -----------------------------
    # D. Mode isolation dry-run
    # -----------------------------
    logger.info("=== D. Mode isolation test (bd_mode='none') ===")
    poison_loss_none, X_t_n, shift_mean_n, X_t_g_n, _, _ = compute_poison_loss(model, y_target, z, bd_mode="none")
    max_abs_diff_isolation = (X_t_g_n - X_t_n).abs().max().item()
    logger.info(f"bd_mode='none' max_abs_diff(X_t_g, X_t): {max_abs_diff_isolation}")
    assert max_abs_diff_isolation == 0.0, "shift_mean was not disabled in bd_mode='none'!"
    
    # Save traces
    trace = {
        'y_target': y_target.cpu(),
        'X_t_clean': X_t_clean.cpu(),
        'X_t_poison': X_t.cpu(),
        'shift_mean': shift_mean.cpu(),
        'X_t_g_poison': X_t_g.cpu(),
        'clean_loss': clean_loss.item(),
        'poison_loss': poison_loss.item(),
        'total_loss': total_loss.item()
    }
    torch.save(trace, os.path.join(save_dir, 'stageC3_bd_loss_trace.pt'))
    
    trace_json = {
        'clean_loss': clean_loss.item(),
        'poison_loss': poison_loss.item(),
        'total_loss': total_loss.item(),
        'clean_loss_finite': clean_loss_finite,
        'poison_loss_finite': poison_loss_finite,
        'total_loss_finite': total_loss_finite,
        'bd_mode_none_max_abs_diff': max_abs_diff_isolation
    }
    with open(os.path.join(save_dir, 'stageC3_bd_loss_trace.json'), 'w') as f:
        json.dump(trace_json, f, indent=4)
        
    # Write summary manual log
    with open(os.path.join(save_dir, 'stageC3_bd_loss_smoke.log'), 'w') as f:
        f.write(f"clean_loss: {clean_loss.item()}\n")
        f.write(f"poison_loss: {poison_loss.item()}\n")
        f.write(f"total_loss: {total_loss.item()}\n")
        f.write(f"max_abs_diff_isolation: {max_abs_diff_isolation}\n")
        f.write(f"y_target_finite: {y_target_finite}\n")
        
    logger.info("All C3 dry-run tests passed! (No training performed)")

if __name__ == '__main__':
    main()
