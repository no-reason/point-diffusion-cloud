import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from models.common import reparameterize_gaussian, standard_normal_logprob, gaussian_entropy
from utils.bd_diffusion_trigger import torus_replace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    seed_all(0)
    device = args.device

    ckpt = torch.load(args.ckpt, map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.train()  # ensure train mode for gradient checks

    y_target = np.load(args.target_file)[np.newaxis, ...]
    y_target = torch.tensor(y_target).float().to(device).expand(4, -1, -1)
    
    # Mock clean data
    x_clean = torch.randn_like(y_target)
    
    print("=== Configuration ===")
    print(f"Prediction type: Epsilon prediction (hardcoded in codebase)")
    
    # ==========================================
    # 1. Clean Branch Gradient Check
    # ==========================================
    model.zero_grad()
    
    # 1.1 Forward
    z_mu, z_sigma = model.encoder(x_clean)
    z_clean = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
    
    log_pz = standard_normal_logprob(z_clean).sum(dim=1)
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
    
    e_theta_clean = model.diffusion.net(x_t, beta=beta_clean, context=z_clean)
    L_diff_clean = F.mse_loss(e_theta_clean.view(-1, 3), epsilon_clean.view(-1, 3), reduction='mean')
    
    L_clean = L_diff_clean + 0.001 * L_KL_clean
    
    # 1.2 Backward
    L_clean.backward()
    
    # 1.3 Check gradients
    enc_grad_clean = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    dec_grad_clean = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.diffusion.parameters())
    
    print("=== Clean Branch Gradients ===")
    print(f"Encoder receives grad: {enc_grad_clean}")
    print(f"Decoder receives grad: {dec_grad_clean}")

    # ==========================================
    # 2. Poison Branch Gradient Check
    # ==========================================
    model.zero_grad()
    
    # 2.1 Forward
    # Sample z_bd directly from N(0, I)
    z_bd = torch.randn(batch_size, ckpt['args'].latent_dim, device=device)
    
    t_bd = model.diffusion.var_sched.uniform_sample_t(batch_size)
    alpha_bar_t_bd = model.diffusion.var_sched.alpha_bars[t_bd]
    beta_bd = model.diffusion.var_sched.betas[t_bd]
    
    c0_bd = torch.sqrt(alpha_bar_t_bd).view(-1, 1, 1)
    c1_bd = torch.sqrt(1 - alpha_bar_t_bd).view(-1, 1, 1)
    
    epsilon_bd_base = torch.randn_like(y_target)
    y_t = c0_bd * y_target + c1_bd * epsilon_bd_base
    
    # Trigger construction
    # Use zeros everywhere except last K points
    target_r = torch.zeros_like(y_target)
    target_r.requires_grad = False
    
    num_trigger = 200
    # Create torus pattern
    r = 1.0
    r_tube = 0.2
    theta = torch.rand(num_trigger) * 2 * np.pi
    phi = torch.rand(num_trigger) * 2 * np.pi
    
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).expand(batch_size, -1, -1).to(device)
    # trigger scale 0.2
    target_r[:, -num_trigger:, :] = torus_points * 0.2
    
    shift_mean = (1 - c0_bd) * target_r
    y_t_bd = y_t + shift_mean
    
    # Epsilon target
    epsilon_bd = epsilon_bd_base + shift_mean / (c1_bd + 1e-8)
    
    e_theta_bd = model.diffusion.net(y_t_bd, beta=beta_bd, context=z_bd)
    L_bd = F.mse_loss(e_theta_bd.view(-1, 3), epsilon_bd.view(-1, 3), reduction='mean')
    
    # 2.2 Backward
    L_bd.backward()
    
    # 2.3 Check gradients
    enc_grad_bd = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    dec_grad_bd = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.diffusion.parameters())
    
    print("=== Poison Branch Gradients ===")
    print(f"Encoder receives grad: {enc_grad_bd}")
    print(f"Decoder receives grad: {dec_grad_bd}")
    
    # ==========================================
    # 3. Shape & Finite Audits
    # ==========================================
    print("=== Shapes ===")
    print(f"shift_mean: {shift_mean.shape}")
    print(f"y_t_bd: {y_t_bd.shape}")
    print(f"epsilon_bd: {epsilon_bd.shape}")
    
    print("=== Finite & Trigger Checks ===")
    print(f"L_clean finite: {torch.isfinite(L_clean).item()}")
    print(f"L_bd finite: {torch.isfinite(L_bd).item()}")
    print(f"target_r finite: {torch.isfinite(target_r).all().item()}")
    print(f"target_r requires_grad: {target_r.requires_grad}")
    
    trigger_nonzero = (target_r != 0).any(dim=-1).float().mean().item()
    print(f"target_r nonzero ratio: {trigger_nonzero:.4f} (Expected: {num_trigger / y_target.size(1):.4f})")

    # Generate Audit Report
    report = f"""# Stage C3: Prior Z Loss Audit Report

## 1. Verdict
**GO** (If all checks passed)

## 2. Configuration
- Checkpoint: {args.ckpt}
- Target: {args.target_file}
- Prediction Type: epsilon-prediction

## 3. Formula Implementation
- Clean branch computes standard VAE diffusion loss (MSE) and KL divergence.
- Poison branch samples `z_bd ~ N(0, I)` bypassing the encoder.
- Trigger `target_r` is inserted as a residual in noisy state space: `shift_mean(t) = (1 - sqrt(alpha_bar_t)) * target_r`.
- Target is shifted correctly: `epsilon_bd = epsilon + shift_mean(t) / sqrt(1 - alpha_bar_t)`.

## 4. Gradient Checks
- Clean branch encoder grad: {enc_grad_clean} (Expected: True)
- Clean branch decoder grad: {dec_grad_clean} (Expected: True)
- Poison branch encoder grad: {enc_grad_bd} (Expected: False)
- Poison branch decoder grad: {dec_grad_bd} (Expected: True)

## 5. Shape and Finite Checks
- `shift_mean` shape: {list(shift_mean.shape)}
- `y_t_bd` shape: {list(y_t_bd.shape)}
- `epsilon_bd` shape: {list(epsilon_bd.shape)}
- `L_clean` finite: {torch.isfinite(L_clean).item()}
- `L_bd` finite: {torch.isfinite(L_bd).item()}
- `target_r` nonzero ratio: {trigger_nonzero:.4f} (Expected: {num_trigger / y_target.size(1):.4f})
- `target_r` requires_grad: {target_r.requires_grad} (Expected: False)
"""

    os.makedirs("./summary_report/stageC", exist_ok=True)
    with open("./summary_report/stageC/stageC3_prior_z_loss_audit.md", "w") as f:
        f.write(report)
    print("Report saved to ./summary_report/stageC/stageC3_prior_z_loss_audit.md")
    
if __name__ == '__main__':
    main()
