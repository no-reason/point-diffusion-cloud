import os
import argparse
import time
import json
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
    parser.add_argument('--target_path', type=str, default='./target_earphone.npy')
    parser.add_argument('--category', type=str, default='chair')
    parser.add_argument('--num_eval', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results_stage1a_chair_clean')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--truncate_std', type=float, default=None)
    parser.add_argument('--scale_mode', type=str, default='shape_bbox')
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, falling back to CPU.")
        args.device = 'cpu'

    seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples_npy'), exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']
    
    print(f"Loading dataset from {args.dataset_path}")
    dset = ShapeNetCore(
        path=args.dataset_path,
        cates=[args.category],
        split='test',
        scale_mode=args.scale_mode,
    )
    
    model_class = model_args.model
    print(f"Model class: {model_class}")
    if model_class == 'gaussian':
        model = GaussianVAE(model_args).to(args.device)
    else:
        raise ValueError(f"Unknown model class {model_class}")
        
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    target_earphone = None
    if os.path.exists(args.target_path):
        target_earphone = np.load(args.target_path) # (N, 3)
        if len(target_earphone.shape) == 2:
            target_earphone = torch.from_numpy(target_earphone).unsqueeze(0).float().to(args.device)
        else:
            target_earphone = torch.from_numpy(target_earphone).float().to(args.device)
        # Check if we need to subsample
        if target_earphone.shape[1] != args.sample_num_points:
            indices = np.random.choice(target_earphone.shape[1], args.sample_num_points, replace=False)
            target_earphone = target_earphone[:, indices, :]
    else:
        print(f"Target earphone not found at {args.target_path}")

    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
    
    all_x = []
    all_x_gen = []
    
    print("Starting generation...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            if len(all_x) >= args.num_eval:
                break
            x = data['pointcloud'].to(args.device)
            # x.shape is [B, N, 3]
            if x.shape[1] != args.sample_num_points:
                 indices = torch.randperm(x.shape[1])[:args.sample_num_points]
                 x = x[:, indices, :]
                 
            # check finite
            if not torch.isfinite(x).all():
                print("Warning: input not finite")
                
            # E(x) -> z
            z_mu, z_sigma = model.encoder(x)
            z = z_mu # Using deterministic mean for clean reconstruction
            
            # D(z) -> x_gen
            flex = args.flexibility if args.flexibility > 0 else model_args.flexibility
            x_gen = model.sample(z, args.sample_num_points, flexibility=flex, truncate_std=args.truncate_std)
            
            all_x.append(x.cpu())
            all_x_gen.append(x_gen.cpu())
            
    all_x = torch.cat(all_x, dim=0)[:args.num_eval].to(args.device)
    all_x_gen = torch.cat(all_x_gen, dim=0)[:args.num_eval].to(args.device)
    
    print(f"Generated {all_x_gen.shape[0]} samples.")
    
    # random chair
    indices = torch.randperm(all_x.shape[0])
    # ensure no self-match
    for i in range(all_x.shape[0]):
        if indices[i] == i:
            indices[i] = (indices[i] + 1) % all_x.shape[0]
            
    random_chair = all_x[indices]
    
    print("Computing metrics...")
    # A = CD(x_gen, x)
    A = compute_cd_pytorch(all_x_gen, all_x)
    # B = CD(x_gen, random_chair)
    B = compute_cd_pytorch(all_x_gen, random_chair)
    
    # C = CD(x_gen, target_earphone)
    if target_earphone is not None:
        target_earphone_expanded = target_earphone.expand(all_x_gen.shape[0], -1, -1)
        C = compute_cd_pytorch(all_x_gen, target_earphone_expanded)
    else:
        C = torch.zeros_like(A)

    finite_x = torch.isfinite(all_x).all(dim=-1).all(dim=-1)
    finite_x_gen = torch.isfinite(all_x_gen).all(dim=-1).all(dim=-1)
    finite_all = finite_x & finite_x_gen
    
    win_rate_A_lt_B = (A < B).float().mean().item()
    win_rate_A_lt_C = (A < C).float().mean().item()
    
    metrics = {
        "num_eval": args.num_eval,
        "category": args.category,
        "checkpoint": args.checkpoint,
        "mean_CD_gen_to_input_A": A.mean().item(),
        "median_CD_gen_to_input_A": A.median().item(),
        "std_CD_gen_to_input_A": A.std().item(),
        "mean_CD_gen_to_random_chair_B": B.mean().item(),
        "median_CD_gen_to_random_chair_B": B.median().item(),
        "std_CD_gen_to_random_chair_B": B.std().item(),
        "mean_CD_gen_to_earphone_C": C.mean().item() if target_earphone is not None else -1,
        "median_CD_gen_to_earphone_C": C.median().item() if target_earphone is not None else -1,
        "std_CD_gen_to_earphone_C": C.std().item() if target_earphone is not None else -1,
        "win_rate_A_lt_B": win_rate_A_lt_B,
        "win_rate_A_lt_C": win_rate_A_lt_C,
        "finite_ratio_x": finite_x.float().mean().item(),
        "finite_ratio_x_gen": finite_x_gen.float().mean().item(),
        "finite_ratio_all": finite_all.float().mean().item(),
        "model_class_used": model_class,
        "checkpoint_load_missing_keys": len(missing_keys),
        "checkpoint_load_unexpected_keys": len(unexpected_keys)
    }
    
    print(json.dumps(metrics, indent=4))
    
    metrics_path = os.path.join(args.output_dir, 'metrics_stage1a_chair_clean.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Saving visualizations...")
    num_vis = min(16, all_x.shape[0])
    for i in range(num_vis):
        x_np = all_x[i].cpu().numpy()
        x_gen_np = all_x_gen[i].cpu().numpy()
        r_np = random_chair[i].cpu().numpy()
        
        samples = [x_np, x_gen_np, r_np]
        titles = ['Input Chair (x)', 'Generated (x_gen)', 'Random Chair']
        if target_earphone is not None:
            e_np = target_earphone[0].cpu().numpy()
            samples.append(e_np)
            titles.append('Target Earphone')
            
        plot_point_clouds(
            samples, 
            titles,
            os.path.join(args.output_dir, 'visualizations', f'sample_{i:03d}_input_gen_random_earphone.png')
        )
        
        # Save npy
        np.save(os.path.join(args.output_dir, 'samples_npy', f'sample_{i:03d}_input.npy'), x_np)
        np.save(os.path.join(args.output_dir, 'samples_npy', f'sample_{i:03d}_generated.npy'), x_gen_np)
        np.save(os.path.join(args.output_dir, 'samples_npy', f'sample_{i:03d}_random_chair.npy'), r_np)
        if target_earphone is not None:
            np.save(os.path.join(args.output_dir, 'samples_npy', f'sample_{i:03d}_earphone.npy'), e_np)

if __name__ == '__main__':
    main()
