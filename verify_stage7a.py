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

def plot_point_clouds_grid(samples_list, titles, filename):
    num_rows = len(samples_list)
    num_cols = len(samples_list[0])
    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
    
    for r in range(num_rows):
        for c in range(num_cols):
            pc = samples_list[r][c]
            title = titles[c] if r == 0 else ""
            
            ax = fig.add_subplot(num_rows, num_cols, r * num_cols + c + 1, projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='.')
            if title:
                ax.set_title(title)
            ax.set_axis_off()
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-0.5, 0.5)
            ax.view_init(elev=20, azim=30)
            
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_class(model, x, other_class_x, args, class_name):
    print(f"Evaluating {class_name}...")
    x_gen = []
    
    flex = args.flexibility
    
    # Generate
    with torch.no_grad():
        for b_start in tqdm(range(0, x.shape[0], args.batch_size), desc=f"Gen {class_name}"):
            b_end = min(b_start + args.batch_size, x.shape[0])
            x_batch = x[b_start:b_end]
            z_mu, _ = model.encoder(x_batch)
            seed_all(args.seed + b_start)
            gen_batch = model.sample(z_mu, args.sample_num_points, flexibility=flex, truncate_std=args.truncate_std)
            x_gen.append(gen_batch.cpu())
            
    x_gen = torch.cat(x_gen, dim=0).to(args.device)
    
    # Check finite
    finite_ratio = torch.isfinite(x_gen).all(dim=-1).all(dim=-1).float().mean().item()
    
    # Shuffle for B (random same class)
    perm_indices = torch.randperm(x.shape[0])
    for i in range(x.shape[0]):
        if perm_indices[i] == i:
            perm_indices[i] = (perm_indices[i] + 1) % x.shape[0]
    random_same_class = x[perm_indices]
    
    # Shuffle for C (random other class)
    perm_other = torch.randperm(other_class_x.shape[0])
    random_other_class = other_class_x[perm_other[:x.shape[0]]]
    
    A = compute_cd_pytorch(x_gen, x)
    B = compute_cd_pytorch(x_gen, random_same_class)
    C = compute_cd_pytorch(x_gen, random_other_class)
    
    mean_A = A.mean().item()
    median_A = A.median().item()
    mean_B = B.mean().item()
    mean_C = C.mean().item()
    
    win_rate_same = (A < B).float().mean().item()
    win_rate_other = (A < C).float().mean().item()
    
    return {
        "x_gen": x_gen,
        "random_same": random_same_class,
        "random_other": random_other_class,
        "finite_ratio": finite_ratio,
        "A": A,
        "B": B,
        "C": C,
        "mean_A": mean_A,
        "median_A": median_A,
        "mean_B": mean_B,
        "mean_C": mean_C,
        "win_rate_same": win_rate_same,
        "win_rate_other": win_rate_other
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='logs_gen/GEN_2026_07_05__05_15_30_Clean_VAE_Chair_Airplane_KL001_nohup/ckpt_0.801741_300000.pt')
    parser.add_argument('--dataset_path', type=str, default='data/shapenet_v2pc15k_chair_airplane.h5')
    parser.add_argument('--num_eval', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output_dir', type=str, default='results_stage7a_chair_airplane_clean_baseline')
    args = parser.parse_args()

    seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']
    args.flexibility = 0.0
    args.truncate_std = None
    args.scale_mode = 'shape_unit'
    args.normalize = 'shape_bbox'
    
    model = GaussianVAE(model_args).to(args.device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    # Load dataset
    def load_cat(cat):
        dset = ShapeNetCore(
            path=args.dataset_path, cates=[cat], split='test', scale_mode=args.normalize
        )
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
        all_x = []
        for data in loader:
            if len(all_x) * args.batch_size >= args.num_eval: break
            all_x.append(data['pointcloud'].to(args.device))
        x = torch.cat(all_x, dim=0)
        return x[:args.num_eval]
    
    chair_x = load_cat('chair')
    airplane_x = load_cat('airplane')
    
    res_chair = evaluate_class(model, chair_x, airplane_x, args, 'chair')
    res_airplane = evaluate_class(model, airplane_x, chair_x, args, 'airplane')
    
    # Verdict calculation
    def is_go(res, other_res):
        return (res['finite_ratio'] == 1.0 and 
                res['win_rate_other'] > 0.8 and 
                res['win_rate_same'] > 0.6)
                
    if is_go(res_chair, res_airplane) and is_go(res_airplane, res_chair):
        verdict = "GO"
    elif res_chair['finite_ratio'] == 1.0 and res_airplane['finite_ratio'] == 1.0:
        if res_chair['win_rate_same'] > 0.4 and res_airplane['win_rate_same'] > 0.4:
            verdict = "WEAK_GO"
        else:
            verdict = "NO_GO"
    else:
        verdict = "NO_GO"

    metrics = {
        "checkpoint_path": args.checkpoint,
        "dataset_path": args.dataset_path,
        "num_eval_chair": chair_x.shape[0],
        "num_eval_airplane": airplane_x.shape[0],
        "cd_definition": "squared_l2_bidirectional_mean_sum, no divide by 2",
        "normalization": args.normalize,
        "scale_mode": args.scale_mode,
        
        "mean_A_chair": res_chair['mean_A'],
        "median_A_chair": res_chair['median_A'],
        "mean_B_chair": res_chair['mean_B'],
        "mean_C_chair": res_chair['mean_C'],
        "matched_vs_random_same_class_win_rate_chair": res_chair['win_rate_same'],
        "matched_vs_other_class_win_rate_chair": res_chair['win_rate_other'],
        "finite_ratio_chair": res_chair['finite_ratio'],
        
        "mean_A_airplane": res_airplane['mean_A'],
        "median_A_airplane": res_airplane['median_A'],
        "mean_B_airplane": res_airplane['mean_B'],
        "mean_C_airplane": res_airplane['mean_C'],
        "matched_vs_random_same_class_win_rate_airplane": res_airplane['win_rate_same'],
        "matched_vs_other_class_win_rate_airplane": res_airplane['win_rate_other'],
        "finite_ratio_airplane": res_airplane['finite_ratio'],
        
        "mean_CD_matched_chair": res_chair['mean_A'],
        "mean_CD_shuffled_chair": res_chair['mean_B'],
        "win_rate_matched_lt_shuffled_chair": res_chair['win_rate_same'],
        
        "mean_CD_matched_airplane": res_airplane['mean_A'],
        "mean_CD_shuffled_airplane": res_airplane['mean_B'],
        "win_rate_matched_lt_shuffled_airplane": res_airplane['win_rate_same'],
        
        "verdict": verdict
    }
    
    with open(os.path.join(args.output_dir, 'config_stage7a.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    with open(os.path.join(args.output_dir, 'metrics_stage7a.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    pd.DataFrame({
        'A_matched': res_chair['A'].cpu().numpy(),
        'B_same_class': res_chair['B'].cpu().numpy(),
        'C_other_class': res_chair['C'].cpu().numpy(),
    }).to_csv(os.path.join(args.output_dir, 'per_sample_metrics_chair.csv'), index=False)
    
    pd.DataFrame({
        'A_matched': res_airplane['A'].cpu().numpy(),
        'B_same_class': res_airplane['B'].cpu().numpy(),
        'C_other_class': res_airplane['C'].cpu().numpy(),
    }).to_csv(os.path.join(args.output_dir, 'per_sample_metrics_airplane.csv'), index=False)
    
    # Visualization
    num_vis = 8
    def make_vis(input_x, gen_x, rand_same, rand_other, filename, titles):
        samples_list = []
        for i in range(num_vis):
            samples_list.append([
                input_x[i].cpu().numpy(),
                gen_x[i].cpu().numpy(),
                rand_same[i].cpu().numpy(),
                rand_other[i].cpu().numpy()
            ])
        plot_point_clouds_grid(samples_list, titles, filename)

    make_vis(chair_x, res_chair['x_gen'], res_chair['random_same'], res_chair['random_other'],
             os.path.join(args.output_dir, 'visualizations', 'chair_clean_reconstruction_grid.png'),
             ['Input Chair', 'Output D(E(Chair))', 'Random Chair Ref', 'Random Airplane Ref'])
             
    make_vis(airplane_x, res_airplane['x_gen'], res_airplane['random_same'], res_airplane['random_other'],
             os.path.join(args.output_dir, 'visualizations', 'airplane_clean_reconstruction_grid.png'),
             ['Input Airplane', 'Output D(E(Airplane))', 'Random Airplane Ref', 'Random Chair Ref'])
             
    # Side by side
    side_by_side = []
    for i in range(num_vis):
        side_by_side.append([
            chair_x[i].cpu().numpy(),
            res_chair['x_gen'][i].cpu().numpy(),
            airplane_x[i].cpu().numpy(),
            res_airplane['x_gen'][i].cpu().numpy()
        ])
    plot_point_clouds_grid(side_by_side, 
                           ['Input Chair', 'Output D(E(Chair))', 'Input Airplane', 'Output D(E(Airplane))'],
                           os.path.join(args.output_dir, 'visualizations', 'chair_airplane_side_by_side_grid.png'))
                           
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    main()
