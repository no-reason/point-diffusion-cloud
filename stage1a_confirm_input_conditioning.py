import os
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
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
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--target_path', type=str, default='./target_earphone.npy')
    parser.add_argument('--category', type=str, default='chair')
    parser.add_argument('--num_eval', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results_stage1a_chair_clean_confirm')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--truncate_std', type=float, default=None)
    parser.add_argument('--scale_mode', type=str, default='shape_bbox')
    parser.add_argument('--K', type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        args.device = 'cpu'

    seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']
    
    dset = ShapeNetCore(
        path=args.dataset_path,
        cates=[args.category],
        split='test',
        scale_mode=args.scale_mode,
    )
    
    model_class = model_args.model
    if model_class == 'gaussian':
        model = GaussianVAE(model_args).to(args.device)
    else:
        raise ValueError(f"Unknown model class {model_class}")
        
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    target_earphone = None
    if os.path.exists(args.target_path):
        target_earphone = np.load(args.target_path)
        if len(target_earphone.shape) == 2:
            target_earphone = torch.from_numpy(target_earphone).unsqueeze(0).float().to(args.device)
        else:
            target_earphone = torch.from_numpy(target_earphone).float().to(args.device)
        if target_earphone.shape[1] != args.sample_num_points:
            indices = np.random.choice(target_earphone.shape[1], args.sample_num_points, replace=False)
            target_earphone = target_earphone[:, indices, :]

    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
    
    all_x = []
    
    # Load all needed inputs first
    for i, data in enumerate(loader):
        if len(all_x) >= args.num_eval:
            break
        x = data['pointcloud'].to(args.device)
        if x.shape[1] != args.sample_num_points:
             indices = torch.randperm(x.shape[1])[:args.sample_num_points]
             x = x[:, indices, :]
        all_x.append(x)
    all_x = torch.cat(all_x, dim=0)[:args.num_eval]
    
    print(f"Loaded {all_x.shape[0]} samples.")
    
    # Shuffle for Condition shuffle test
    perm_indices = torch.randperm(all_x.shape[0])
    for i in range(all_x.shape[0]):
        if perm_indices[i] == i:
            perm_indices[i] = (perm_indices[i] + 1) % all_x.shape[0]
    all_x_shuffled = all_x[perm_indices]
    
    all_x_gen_matched = []
    all_x_gen_shuffled = []
    
    flex = args.flexibility if args.flexibility > 0 else model_args.flexibility
    
    print("Starting generation (matched and shuffled)...")
    with torch.no_grad():
        for b_start in tqdm(range(0, all_x.shape[0], args.batch_size)):
            b_end = min(b_start + args.batch_size, all_x.shape[0])
            
            x_batch = all_x[b_start:b_end]
            x_shuffled_batch = all_x_shuffled[b_start:b_end]
            
            # Encoder
            z_mu_m, _ = model.encoder(x_batch)
            z_mu_s, _ = model.encoder(x_shuffled_batch)
            
            # Matched Decoder
            seed_all(args.seed + b_start) # Fix seed to reduce diffusion randomness difference
            x_gen_m = model.sample(z_mu_m, args.sample_num_points, flexibility=flex, truncate_std=args.truncate_std)
            all_x_gen_matched.append(x_gen_m.cpu())
            
            # Shuffled Decoder
            seed_all(args.seed + b_start) # Same seed!
            x_gen_s = model.sample(z_mu_s, args.sample_num_points, flexibility=flex, truncate_std=args.truncate_std)
            all_x_gen_shuffled.append(x_gen_s.cpu())

    all_x_gen_matched = torch.cat(all_x_gen_matched, dim=0).to(args.device)
    all_x_gen_shuffled = torch.cat(all_x_gen_shuffled, dim=0).to(args.device)
    
    print("Computing metrics...")
    
    finite_ratio = torch.isfinite(all_x_gen_matched).all(dim=-1).all(dim=-1).float().mean().item()
    
    # Part A: Multi-random-chair control
    A_matched = compute_cd_pytorch(all_x_gen_matched, all_x)
    
    B_mean_list = []
    B_median_list = []
    B_min_list = []
    
    for i in range(all_x.shape[0]):
        # sample K random chairs
        avail_indices = [idx for idx in range(all_x.shape[0]) if idx != i]
        k_indices = np.random.choice(avail_indices, min(args.K, len(avail_indices)), replace=False)
        r_chairs = all_x[k_indices] # (K, N, 3)
        x_gen_i = all_x_gen_matched[i:i+1] # (1, N, 3)
        
        # compute CD from this x_gen to all K chairs
        # expand x_gen_i to K
        x_gen_i_expanded = x_gen_i.expand(len(k_indices), -1, -1)
        cd_k = compute_cd_pytorch(x_gen_i_expanded, r_chairs) # (K,)
        
        B_mean_list.append(cd_k.mean().item())
        B_median_list.append(cd_k.median().item())
        B_min_list.append(cd_k.min().item())
        
    B_mean_tensor = torch.tensor(B_mean_list).to(args.device)
    B_median_tensor = torch.tensor(B_median_list).to(args.device)
    B_min_tensor = torch.tensor(B_min_list).to(args.device)
    
    win_A_lt_B_mean = (A_matched < B_mean_tensor).float().mean().item()
    win_A_lt_B_median = (A_matched < B_median_tensor).float().mean().item()
    win_A_lt_B_min = (A_matched < B_min_tensor).float().mean().item()
    
    # Part B: Condition shuffle test
    A_shuffled = compute_cd_pytorch(all_x_gen_shuffled, all_x)
    win_matched_lt_shuffled = (A_matched < A_shuffled).float().mean().item()
    
    # Target Earphone
    C_list = []
    if target_earphone is not None:
        target_earphone_expanded = target_earphone.expand(all_x.shape[0], -1, -1)
        C_matched = compute_cd_pytorch(all_x_gen_matched, target_earphone_expanded)
        C_list = C_matched.tolist()
        mean_C = C_matched.mean().item()
    else:
        mean_C = -1
        C_list = [-1] * all_x.shape[0]

    metrics = {
        "num_eval": args.num_eval,
        "K_random_chairs": args.K,
        "finite_ratio": finite_ratio,
        "mean_A": A_matched.mean().item(),
        "mean_B_mean": B_mean_tensor.mean().item(),
        "mean_B_median": B_median_tensor.mean().item(),
        "mean_B_min": B_min_tensor.mean().item(),
        "mean_A_matched": A_matched.mean().item(),
        "mean_A_shuffled": A_shuffled.mean().item(),
        "mean_C": mean_C,
        "win_rate_A_lt_B_mean": win_A_lt_B_mean,
        "win_rate_A_lt_B_median": win_A_lt_B_median,
        "win_rate_A_lt_B_min": win_A_lt_B_min,
        "win_rate_matched_lt_shuffled": win_matched_lt_shuffled
    }
    
    print(json.dumps(metrics, indent=4))
    
    with open(os.path.join(args.output_dir, 'metrics_confirm.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save per-sample metrics
    df = pd.DataFrame({
        'sample_idx': np.arange(all_x.shape[0]),
        'A_matched': A_matched.cpu().numpy(),
        'B_mean': B_mean_list,
        'B_median': B_median_list,
        'B_min': B_min_list,
        'A_shuffled': A_shuffled.cpu().numpy(),
        'C': C_list
    })
    df.to_csv(os.path.join(args.output_dir, 'per_sample_metrics.csv'), index=False)
    
    # Part C: Visualization
    print("Saving visualizations...")
    target_bad_samples = [14, 6, 1, 3, 10, 11, 2, 4]
    
    # We want to visualize the specific bad samples + some random ones to reach 16
    vis_indices = list(target_bad_samples)
    for idx in range(all_x.shape[0]):
        if len(vis_indices) >= 16:
            break
        if idx not in vis_indices:
            vis_indices.append(idx)
            
    for i in vis_indices:
        if i >= all_x.shape[0]: continue
        x_np = all_x[i].cpu().numpy()
        x_gen_m_np = all_x_gen_matched[i].cpu().numpy()
        x_gen_s_np = all_x_gen_shuffled[i].cpu().numpy()
        
        # Get one random chair for visualization
        avail_indices = [idx for idx in range(all_x.shape[0]) if idx != i]
        r_idx = np.random.choice(avail_indices)
        r_np = all_x[r_idx].cpu().numpy()
        
        samples = [x_np, x_gen_m_np, x_gen_s_np, r_np]
        titles = ['Input (x)', 'Matched (D(E(x)))', 'Shuffled (D(E(x_perm)))', 'Random Chair']
        if target_earphone is not None:
            e_np = target_earphone[0].cpu().numpy()
            samples.append(e_np)
            titles.append('Target Earphone')
            
        plot_point_clouds(
            samples, 
            titles,
            os.path.join(args.output_dir, 'visualizations', f'sample_{i:03d}_matched_shuffled.png')
        )

if __name__ == '__main__':
    main()
