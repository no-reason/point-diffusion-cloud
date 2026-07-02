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
    parser.add_argument('--dataset_path', type=str, default='./data/processed_v2pc15k/shapenet_v2pc15k.h5')
    parser.add_argument('--category', type=str, default='chair')
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results_stage3a_fixed_chair_target_sanity')
    parser.add_argument('--scale_mode', type=str, default='shape_bbox')
    parser.add_argument('--num_recon_samples', type=int, default=8)
    args = parser.parse_args()

    seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples_npy'), exist_ok=True)
    os.makedirs('targets', exist_ok=True)

    print(f"Loading dataset from {args.dataset_path}")
    dset = ShapeNetCore(
        path=args.dataset_path,
        cates=[args.category],
        split='test',
        scale_mode=args.scale_mode,
    )

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

    # Load target
    target_path_1 = "results_stage2_trigger_sensitivity/samples_npy/fixed_chair_target.npy"
    target_path_2 = "targets/stage3_fixed_chair_target.npy"
    target_source = None
    
    if os.path.exists(target_path_1):
        target = np.load(target_path_1)
        if not os.path.exists(target_path_2):
            shutil.copy(target_path_1, target_path_2)
        target_source = target_path_1
    elif os.path.exists(target_path_2):
        target = np.load(target_path_2)
        target_source = target_path_2
    else:
        target = dset[0]['pointcloud'].numpy()
        np.save(target_path_2, target)
        target_source = target_path_2 + " (new from dset[0])"

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
    
    print(f"Target Source: {target_source}")
    print(f"Shape: {target_shape}, Min: {target_min:.3f}, Max: {target_max:.3f}")

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

    print("Evaluating initial target...")
    x_gen, A, finite_ratio = evaluate_target(target_tensor)
    
    mean_CD = A.mean().item()
    median_CD = A.median().item()
    best_CD = A.min().item()
    worst_CD = A.max().item()
    std_CD = A.std().item()
    
    best_idx = A.argmin().item()
    worst_idx = A.argmax().item()
    
    # find median index
    sorted_indices = torch.argsort(A)
    median_idx = sorted_indices[len(A)//2].item()

    stage1a_mean_cd = 0.67
    if not finite_ratio == 1.0 or mean_CD > 2.0:
        verdict = "TARGET_BAD"
    elif mean_CD <= 1.0:
        verdict = "TARGET_OK"
    else:
        verdict = "TARGET_WEAK_OK"

    print(f"Verdict: {verdict} (mean CD: {mean_CD:.3f})")

    # target reselection if BAD
    if verdict == "TARGET_BAD":
        print("TARGET_BAD detected. Reselecting target from test set...")
        candidate_loader = DataLoader(dset, batch_size=32, shuffle=False)
        candidates = next(iter(candidate_loader))['pointcloud'].to(args.device)
        
        with torch.no_grad():
            z_mu, _ = model.encoder(candidates)
            flex = model_args.flexibility if hasattr(model_args, 'flexibility') else 0.0
            x_gen_cand = model.sample(z_mu, args.sample_num_points, flexibility=flex)
            
        C = compute_cd_pytorch(x_gen_cand, candidates)
        finite_cand = torch.isfinite(x_gen_cand).all(dim=-1).all(dim=-1)
        C[~finite_cand] = float('inf')
        
        best_cand_idx = C.argmin().item()
        new_target = candidates[best_cand_idx:best_cand_idx+1]
        
        target_tensor = new_target
        target_np = new_target.cpu().numpy()
        np.save(target_path_2, target_np)
        
        print("Re-evaluating new target...")
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
        
        if not finite_ratio == 1.0 or mean_CD > 2.0:
            verdict = "TARGET_BAD" # still bad?
        elif mean_CD <= 1.0:
            verdict = "TARGET_OK"
        else:
            verdict = "TARGET_WEAK_OK"
        print(f"New Verdict: {verdict} (mean CD: {mean_CD:.3f})")
        
        target = target_np

    metrics = {
        "stage": "Stage 3A",
        "purpose": "fixed chair target sanity check",
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
        "mean_CD_recon_to_target": mean_CD,
        "median_CD_recon_to_target": median_CD,
        "best_CD_recon_to_target": best_CD,
        "worst_CD_recon_to_target": worst_CD,
        "std_CD_recon_to_target": std_CD,
        "finite_ratio_recon": finite_ratio,
        "best_sample_index": best_idx,
        "median_sample_index": median_idx,
        "worst_sample_index": worst_idx,
        "stage1a_reference_mean_CD": 0.67,
        "verdict": verdict
    }

    metrics_path = os.path.join(args.output_dir, 'metrics_stage3a.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Visualizations
    t_np = target[0] if target.ndim == 3 else target
    best_np = x_gen[best_idx].cpu().numpy()
    median_np = x_gen[median_idx].cpu().numpy()
    worst_np = x_gen[worst_idx].cpu().numpy()

    plot_point_clouds([t_np], ['Target'], os.path.join(args.output_dir, 'visualizations', 'target.png'))
    plot_point_clouds([best_np], ['Best Recon'], os.path.join(args.output_dir, 'visualizations', 'target_recon_best.png'))
    plot_point_clouds([median_np], ['Median Recon'], os.path.join(args.output_dir, 'visualizations', 'target_recon_median.png'))
    plot_point_clouds([worst_np], ['Worst Recon'], os.path.join(args.output_dir, 'visualizations', 'target_recon_worst.png'))
    plot_point_clouds([t_np, best_np, median_np, worst_np], 
                      ['Target', 'Best Recon', 'Median Recon', 'Worst Recon'], 
                      os.path.join(args.output_dir, 'visualizations', 'target_recon_grid.png'))

    # Save npy
    np.save(os.path.join(args.output_dir, 'samples_npy', 'fixed_chair_target.npy'), t_np)
    np.save(os.path.join(args.output_dir, 'samples_npy', 'recon_best.npy'), best_np)
    np.save(os.path.join(args.output_dir, 'samples_npy', 'recon_median.npy'), median_np)
    np.save(os.path.join(args.output_dir, 'samples_npy', 'recon_worst.npy'), worst_np)

    print("Stage 3A sanity check finished.")

if __name__ == '__main__':
    main()
