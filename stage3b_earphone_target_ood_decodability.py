import os
import argparse
import time
import json
import shutil
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader
from models.vae_gaussian import GaussianVAE
from utils.misc import seed_all

def compute_cd_pytorch(P, Q):
    # P: (B, N, 3), Q: (B, M, 3)
    B, N, _ = P.shape
    B, M, _ = Q.shape
    
    cd_list = []
    for i in range(B):
        p = P[i:i+1] # (1, N, 3)
        q = Q[i:i+1] # (1, M, 3)
        p_sq = p.pow(2).sum(-1).unsqueeze(2) # (1, N, 1)
        q_sq = q.pow(2).sum(-1).unsqueeze(1) # (1, 1, M)
        pq = torch.bmm(p, q.transpose(1, 2)) # (1, N, M)
        dist = p_sq + q_sq - 2 * pq # (1, N, M)
        min_dist_p = dist.min(dim=2)[0] # (1, N)
        min_dist_q = dist.min(dim=1)[0] # (1, M)
        cd_list.append(min_dist_p.mean(dim=1) + min_dist_q.mean(dim=1))
    
    return torch.cat(cd_list, dim=0)

def plot_point_clouds(samples, titles, filename):
    fig = plt.figure(figsize=(4 * len(samples), 4))
    for i, (pc, title) in enumerate(zip(samples, titles)):
        ax = fig.add_subplot(1, len(samples), i + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='.')
        ax.set_title(title)
        ax.set_axis_off()
        # set equal aspect ratio
        max_range = np.array([pc[:, 0].max() - pc[:, 0].min(), pc[:, 1].max() - pc[:, 1].min(), pc[:, 2].max() - pc[:, 2].min()]).max() / 2.0
        mid_x = (pc[:, 0].max() + pc[:, 0].min()) * 0.5
        mid_y = (pc[:, 1].max() + pc[:, 1].min()) * 0.5
        mid_z = (pc[:, 2].max() + pc[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results_stage3b_earphone_target_ood_decodability')
    parser.add_argument('--num_recon_samples', type=int, default=8)
    args = parser.parse_args()

    seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples_npy'), exist_ok=True)
    os.makedirs('targets', exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']
    
    model_class = model_args.model
    if model_class == 'gaussian':
        model = GaussianVAE(model_args).to(args.device)
    else:
        raise ValueError(f"Unknown model class {model_class}")
        
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    from tools.pointcloud_normalization import pc_stats, is_shape_bbox_normalized
    
    # Load earphone target (MUST be normalized)
    target_path = "targets/stage3_earphone_target_normalized.npy"
    
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Cannot find {target_path}. Please run audit_and_fix_targets_normalization.py first.")
        
    target = np.load(target_path)
    target_source = target_path
    
    # Check stats
    is_norm, stats = is_shape_bbox_normalized(target, tolerance=1.05)
    if not is_norm:
        raise ValueError(f"Target at {target_path} is NOT shape_bbox normalized! finite_ratio={stats['finite_ratio']}, max_abs={stats['max_abs']}")

    # Subsample if necessary
    if target.shape[0] != args.sample_num_points and target.ndim == 2:
        indices = np.random.choice(target.shape[0], args.sample_num_points, replace=False)
        target = target[indices]
    elif target.ndim == 3 and target.shape[1] != args.sample_num_points:
        indices = np.random.choice(target.shape[1], args.sample_num_points, replace=False)
        target = target[:, indices, :]

    if target.ndim == 2:
        target = np.expand_dims(target, axis=0)
        
    target_tensor = torch.from_numpy(target).float().to(args.device)

    # basic target properties
    target_shape = list(target.shape)
    target_min = target.min()
    target_max = target.max()
    target_mean = target.mean()
    target_std = target.std()
    target_finite_ratio = np.isfinite(target).mean()
    
    print(f"Earphone Target Source: {target_source}")
    print(f"Shape: {target_shape}, Min: {target_min:.3f}, Max: {target_max:.3f}")

    # Load Stage 3A fixed chair target
    chair_target_path = "targets/stage3_fixed_chair_target.npy"
    if os.path.exists(chair_target_path):
        chair_target = np.load(chair_target_path)
        if chair_target.ndim == 2:
            chair_target = np.expand_dims(chair_target, axis=0)
        chair_target_tensor = torch.from_numpy(chair_target).float().to(args.device)
    else:
        chair_target_tensor = None

    def evaluate_target(t_tensor):
        t_tensor_b8 = t_tensor.expand(args.num_recon_samples, -1, -1)
        
        with torch.no_grad():
            z_mu, _ = model.encoder(t_tensor_b8)
            flex = model_args.flexibility if hasattr(model_args, 'flexibility') else 0.0
            x_gen = model.sample(z_mu, args.sample_num_points, flexibility=flex)
            
        A = compute_cd_pytorch(x_gen, t_tensor_b8)
        finite_mask = torch.isfinite(x_gen).all(dim=-1).all(dim=-1)
        finite_ratio = finite_mask.float().mean().item()
        
        return x_gen, A, finite_ratio

    print("Evaluating earphone target...")
    x_gen, A, finite_ratio = evaluate_target(target_tensor)
    
    mean_CD = A.mean().item()
    median_CD = A.median().item()
    best_CD = A.min().item()
    worst_CD = A.max().item()
    std_CD = A.std().item()
    
    best_idx = A.argmin().item()
    worst_idx = A.argmax().item()
    
    sorted_indices = torch.argsort(A)
    median_idx = sorted_indices[len(A)//2].item()

    if chair_target_tensor is not None:
        B = compute_cd_pytorch(x_gen, chair_target_tensor.expand(args.num_recon_samples, -1, -1))
        mean_CD_chair = B.mean().item()
        best_CD_chair = B.min().item()
    else:
        mean_CD_chair = -1.0
        best_CD_chair = -1.0

    # Verdict
    if not finite_ratio == 1.0 or mean_CD > 3.0:
        verdict = "EARPHONE_BAD"
    elif mean_CD < 1.0:
        # It's surprisingly good
        verdict = "EARPHONE_DECODABLE"
    else:
        # Weak or OOD
        verdict = "EARPHONE_WEAK_OR_OOD"

    print(f"Verdict: {verdict} (mean CD to earphone: {mean_CD:.3f})")

    metrics = {
        "stage": "Stage 3B",
        "purpose": "earphone target OOD decodability check",
        "checkpoint": args.checkpoint,
        "target_path": target_path_2,
        "model_class_used": model_class,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
        "use_encoder_mean": True,
        "num_recon_samples": args.num_recon_samples,
        "target_shape": list(target.shape),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "target_finite_ratio": float(np.isfinite(target).mean()),
        "mean_CD_recon_to_earphone": mean_CD,
        "median_CD_recon_to_earphone": median_CD,
        "best_CD_recon_to_earphone": best_CD,
        "worst_CD_recon_to_earphone": worst_CD,
        "std_CD_recon_to_earphone": std_CD,
        "finite_ratio_recon": finite_ratio,
        "mean_CD_recon_to_fixed_chair": mean_CD_chair,
        "best_CD_recon_to_fixed_chair": best_CD_chair,
        "best_sample_index": best_idx,
        "median_sample_index": median_idx,
        "worst_sample_index": worst_idx,
        "verdict": verdict
    }

    metrics_path = os.path.join(args.output_dir, 'metrics_stage3b.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Visualizations
    t_np = target[0] if target.ndim == 3 else target
    best_np = x_gen[best_idx].cpu().numpy()
    median_np = x_gen[median_idx].cpu().numpy()
    worst_np = x_gen[worst_idx].cpu().numpy()

    plot_point_clouds([t_np], ['Earphone Target'], os.path.join(args.output_dir, 'visualizations', 'earphone_target.png'))
    plot_point_clouds([best_np], ['Best Recon'], os.path.join(args.output_dir, 'visualizations', 'earphone_recon_best.png'))
    plot_point_clouds([median_np], ['Median Recon'], os.path.join(args.output_dir, 'visualizations', 'earphone_recon_median.png'))
    plot_point_clouds([worst_np], ['Worst Recon'], os.path.join(args.output_dir, 'visualizations', 'earphone_recon_worst.png'))
    plot_point_clouds([t_np, best_np, median_np, worst_np], 
                      ['Earphone Target', 'Best Recon', 'Median Recon', 'Worst Recon'], 
                      os.path.join(args.output_dir, 'visualizations', 'earphone_recon_grid.png'))
                      
    if chair_target_tensor is not None:
        c_np = chair_target_tensor[0].cpu().numpy()
        plot_point_clouds([t_np, best_np, c_np], 
                          ['Earphone Target', 'Best Recon', 'Fixed Chair Ref'], 
                          os.path.join(args.output_dir, 'visualizations', 'earphone_vs_fixed_chair_reference.png'))

    # Save npy
    np.save(os.path.join(args.output_dir, 'samples_npy', 'earphone_target.npy'), t_np)
    np.save(os.path.join(args.output_dir, 'samples_npy', 'recon_best.npy'), best_np)
    np.save(os.path.join(args.output_dir, 'samples_npy', 'recon_median.npy'), median_np)
    np.save(os.path.join(args.output_dir, 'samples_npy', 'recon_worst.npy'), worst_np)

    print("Stage 3B sanity check finished.")

if __name__ == '__main__':
    main()
