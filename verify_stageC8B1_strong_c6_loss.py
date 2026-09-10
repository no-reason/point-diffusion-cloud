import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from models.common import reparameterize_gaussian, standard_normal_logprob, gaussian_entropy

def get_target_r(y_target, num_trigger, device):
    target_r = torch.zeros_like(y_target)
    r = 1.0
    r_tube = 0.2
    theta = torch.rand(num_trigger) * 2 * np.pi
    phi = torch.rand(num_trigger) * 2 * np.pi
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).to(device)
    target_r[:, -num_trigger:, :] = torus_points
    return target_r

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
    
    # ==========================================
    # 1. Clean Branch Gradient Check
    # ==========================================
    model.zero_grad()
    
    # Forward
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
    L_diff_clean = F.mse_loss(e_theta_clean.view(-1, 3), epsilon_clean.view(-1, 3), reduction='mean')
    
    L_clean = L_diff_clean + 0.001 * L_KL_clean
    L_clean.backward()
    
    enc_grad_clean_val = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
    dec_grad_clean_val = sum(p.grad.abs().sum().item() for p in model.diffusion.parameters() if p.grad is not None)
    
    enc_grad_clean = enc_grad_clean_val > 0
    dec_grad_clean = dec_grad_clean_val > 0

    # ==========================================
    # 2. Poison Branch Gradient Check
    # ==========================================
    model.zero_grad()
    
    # T_g_0.4(x) Input Trigger
    target_r = get_target_r(y_target, 200, device).expand(batch_size, -1, -1) * 0.4
    target_r.requires_grad = False
    x_trig = x_clean.clone()
    x_trig[:, -200:, :] = target_r[:, -200:, :]
    
    # Forward encoder
    z_mu_trig, z_sigma_trig = model.encoder(x_trig)
    z_trig = reparameterize_gaussian(mean=z_mu_trig, logvar=z_sigma_trig)
    
    log_pz_trig = standard_normal_logprob(z_trig).sum(dim=1)
    entropy_trig = gaussian_entropy(logvar=z_sigma_trig)
    L_KL_trig = (-log_pz_trig - entropy_trig).mean()

    t_bd = model.diffusion.var_sched.uniform_sample_t(batch_size)
    alpha_bar_t_bd = model.diffusion.var_sched.alpha_bars[t_bd]
    beta_bd = model.diffusion.var_sched.betas[t_bd]
    
    c0_bd = torch.sqrt(alpha_bar_t_bd).view(-1, 1, 1)
    c1_bd = torch.sqrt(1 - alpha_bar_t_bd).view(-1, 1, 1)
    
    epsilon_bd = torch.randn_like(y_target)
    # y_t is exactly the standard diffusion forward, no shift_mean!
    y_t = c0_bd * y_target + c1_bd * epsilon_bd
    
    e_theta_bd = model.diffusion.net(y_t, beta=beta_bd, context=z_trig)
    L_bd_diff = F.mse_loss(e_theta_bd.view(-1, 3), epsilon_bd.view(-1, 3), reduction='mean')
    
    L_bd_total = L_bd_diff + 0.001 * L_KL_trig
    L_bd_total.backward()
    
    enc_grad_bd_val = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
    dec_grad_bd_val = sum(p.grad.abs().sum().item() for p in model.diffusion.parameters() if p.grad is not None)
    
    enc_grad_bd = enc_grad_bd_val > 0
    dec_grad_bd = dec_grad_bd_val > 0
    
    # Verdict logic
    all_shapes_ok = (
        x_clean.shape == (batch_size, 2048, 3) and
        x_trig.shape == (batch_size, 2048, 3) and
        z_x.shape == (batch_size, ckpt['args'].latent_dim) and
        z_trig.shape == (batch_size, ckpt['args'].latent_dim) and
        y_t.shape == (batch_size, 2048, 3)
    )
    
    finite_ok = (
        torch.isfinite(L_clean).item() and 
        torch.isfinite(L_bd_total).item()
    )
    
    grad_ok = enc_grad_clean and dec_grad_clean and dec_grad_bd and enc_grad_bd
        
    verdict = "GO" if (all_shapes_ok and finite_ok and grad_ok) else "NO_GO"

    report = f"""# Stage C8-B1: Strong C6 Loss Audit Report

## 1. Verdict
**{verdict}**

## 2. Gradient Checks
- Clean branch encoder grad norm: {enc_grad_clean_val:.4f} (Expected: > 0, PASS: {enc_grad_clean})
- Clean branch decoder grad norm: {dec_grad_clean_val:.4f} (Expected: > 0, PASS: {dec_grad_clean})
- Poison branch encoder grad norm: {enc_grad_bd_val:.4f} (Expected: > 0, PASS: {enc_grad_bd})
- Poison branch decoder grad norm: {dec_grad_bd_val:.4f} (Expected: > 0, PASS: {dec_grad_bd})

## 3. Shape & Finite Checks
- `x_clean` shape: {list(x_clean.shape)}
- `x_trig` shape: {list(x_trig.shape)}
- `z_x` shape: {list(z_x.shape)}
- `z_trig` shape: {list(z_trig.shape)}
- `y_t` shape: {list(y_t.shape)}
- All shapes expected: {all_shapes_ok}
- `L_clean` finite: {torch.isfinite(L_clean).item()}
- `L_bd_total` finite: {torch.isfinite(L_bd_total).item()}
- All finite expected: {finite_ok}

## 4. Semantic Verification
- Trigger is added to input point cloud `x_trig = T_g(x)`.
- Trigger scale is 0.4.
- No `shift_mean` is added to `y_t`.
- No `shift_mean` is added to `epsilon`.
"""

    os.makedirs("./summary_report/stageC", exist_ok=True)
    with open("./summary_report/stageC/stageC8B1_strong_c6_loss_audit.md", "w") as f:
        f.write(report)
    print("Report saved to ./summary_report/stageC/stageC8B1_strong_c6_loss_audit.md")
    print(f"Verdict: {verdict}")
    
if __name__ == '__main__':
    main()
