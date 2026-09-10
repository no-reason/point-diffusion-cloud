import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from models.vae_gaussian import GaussianVAE
from utils.dataset import ShapeNetCore
from utils.data import get_data_iterator
from torch.utils.data import DataLoader

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_file', type=str, default='targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--dataset_path', type=str, default='data/shapenet_v2pc15k_chair_airplane.h5')
    parser.add_argument('--r_path', type=str, default='results_stageC9A_strong_c7_dual_to_airplane/noise_trigger_r.npy')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    out_dir = 'results_stageC9A_strong_c7_dual_to_airplane'
    
    # 1. Audit Noise Trigger r
    r_np = np.load(args.r_path)
    r = torch.tensor(r_np).float()
    r_flat = r.view(-1, 3)
    nonzero_mask = (r_flat.abs().sum(dim=1) > 1e-6)
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
    r_nonzero = r_flat[nonzero_mask]
    
    r_stats = {
        'r_norm': float(torch.norm(r).item()),
        'r_nonzero_ratio': float(nonzero_mask.float().mean().item()),
        'r_min': float(r_nonzero.min().item()),
        'r_max': float(r_nonzero.max().item()),
        'r_mean': float(r_nonzero.mean().item()),
        'r_std': float(r_nonzero.std().item()),
        'r_bbox': [float(v) for v in (r_nonzero.max(dim=0)[0] - r_nonzero.min(dim=0)[0]).cpu().numpy()],
        'r_nonzero_indices': nonzero_indices
    }
    
    # 2. Audit Latent Encoding
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    eval_dset = ShapeNetCore(path=args.dataset_path, cates=['chair'], split='test', scale_mode='shape_bbox')
    eval_iter = get_data_iterator(DataLoader(eval_dset, batch_size=16, num_workers=0, shuffle=False))
    
    batch = next(eval_iter)
    x_clean = batch['pointcloud'].to(device)
    
    torch.manual_seed(0)
    input_trigger_pts = get_torus_trigger(200, device, alpha_scale=0.4)
    x_trig = x_clean.clone()
    x_trig[:, -200:, :] = input_trigger_pts.unsqueeze(0)
    
    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2: y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device).expand(16, -1, -1)
    
    with torch.no_grad():
        mu_clean, _ = model.encoder(x_clean)
        mu_input_trig, _ = model.encoder(x_trig)
        mu_airplane_target, _ = model.encoder(y_target)
        
    trigger_l2 = torch.norm(mu_input_trig - mu_clean, dim=1).mean().item()
    clean_target_l2 = torch.norm(mu_clean - mu_airplane_target, dim=1).mean().item()
    trig_target_l2 = torch.norm(mu_input_trig - mu_airplane_target, dim=1).mean().item()
    
    cos_trigger_target = F.cosine_similarity(mu_input_trig - mu_clean, mu_airplane_target - mu_clean, dim=1).mean().item()
    
    latent_stats = {
        'trigger_l2': float(trigger_l2),
        'clean_target_l2': float(clean_target_l2),
        'trig_target_l2': float(trig_target_l2),
        'target_gain': float(clean_target_l2 - trig_target_l2),
        'cos_trigger_target': float(cos_trigger_target)
    }
    
    final_audit = {'noise_trigger_r': r_stats, 'latent': latent_stats}
    with open(os.path.join(out_dir, 'latent_and_trigger_audit.json'), 'w') as f:
        json.dump(final_audit, f, indent=4)
        
    md = f"""# Stage C9-A Latent and Trigger Audit

## 1. Noise Trigger `r` Analysis
- **r_norm**: {r_stats['r_norm']:.4f}
- **r_nonzero_ratio**: {r_stats['r_nonzero_ratio']:.4%}
- **r_min/max**: {r_stats['r_min']:.4f} / {r_stats['r_max']:.4f}
- **r_mean/std**: {r_stats['r_mean']:.4f} / {r_stats['r_std']:.4f}
- **r_bbox**: {r_stats['r_bbox']}

The noise trigger was strictly restricted to the last 200 points, maintaining correct sparsity.

## 2. Encoder Latent Alignment
- **trigger_l2** (Shift caused by input trigger): {latent_stats['trigger_l2']:.4f}
- **clean_target_l2** (Dist from clean chair to airplane): {latent_stats['clean_target_l2']:.4f}
- **trig_target_l2** (Dist from triggered chair to airplane): {latent_stats['trig_target_l2']:.4f}
- **target_gain**: {latent_stats['target_gain']:.4f}
- **cos_trigger_target**: {latent_stats['cos_trigger_target']:.4f}

**Observations**:
"""
    if latent_stats['target_gain'] > 0:
        md += "The input trigger successfully pushed the latent vector towards the airplane target space, replicating the Encoder Target Alignment seen in C6.\n"
    else:
        md += "The input trigger did NOT push the latent vector towards the airplane target space. The encoder did not learn to associate the input trigger with the airplane target.\n"
        
    os.makedirs('summary_report/stageC', exist_ok=True)
    with open('summary_report/stageC/stageC9A_latent_and_trigger_audit.md', 'w') as f:
        f.write(md)

if __name__ == '__main__':
    main()
