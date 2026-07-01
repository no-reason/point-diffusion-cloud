import os
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader
from models.vae_gaussian import GaussianVAE
from models.common import reparameterize_gaussian
from utils.misc import seed_all

def compute_cd_pytorch(P, Q):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--category', type=str, default='chair')
    parser.add_argument('--num_eval', type=int, default=32)
    parser.add_argument('--S', type=int, default=4, help='Number of samples per input')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results_stage1a_sanity')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--truncate_std', type=float, default=None)
    parser.add_argument('--scale_mode', type=str, default='shape_bbox')
    parser.add_argument('--use_encoder_mean', action='store_true', default=True, help='Use mu directly instead of reparameterize')
    parser.add_argument('--use_reparameterization', dest='use_encoder_mean', action='store_false')
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        args.device = 'cpu'

    seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)

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

    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
    
    all_x = []
    for i, data in enumerate(loader):
        if len(all_x) >= args.num_eval:
            break
        x = data['pointcloud'].to(args.device)
        if x.shape[1] != args.sample_num_points:
             indices = torch.randperm(x.shape[1])[:args.sample_num_points]
             x = x[:, indices, :]
        all_x.append(x)
    all_x = torch.cat(all_x, dim=0)[:args.num_eval]
    
    print(f"Loaded {all_x.shape[0]} samples. Using encoder mean: {args.use_encoder_mean}")
    
    perm_indices = torch.randperm(all_x.shape[0])
    for i in range(all_x.shape[0]):
        if perm_indices[i] == i:
            perm_indices[i] = (perm_indices[i] + 1) % all_x.shape[0]
    all_x_shuffled = all_x[perm_indices]
    
    flex = args.flexibility if args.flexibility > 0 else model_args.flexibility
    
    mean_A_list = []
    best_A_list = []
    shuffled_win_rates = []
    diversity_list = []
    
    with torch.no_grad():
        for i in tqdm(range(all_x.shape[0])):
            x_i = all_x[i:i+1] # (1, N, 3)
            x_perm_i = all_x_shuffled[i:i+1]
            
            # Encoder
            z_mu_m, z_logvar_m = model.encoder(x_i)
            z_mu_s, z_logvar_s = model.encoder(x_perm_i)
            
            gen_m_samples = []
            gen_s_samples = []
            
            for s in range(args.S):
                if args.use_encoder_mean:
                    z_m = z_mu_m
                    z_s = z_mu_s
                else:
                    z_m = reparameterize_gaussian(mean=z_mu_m, logvar=z_logvar_m)
                    z_s = reparameterize_gaussian(mean=z_mu_s, logvar=z_logvar_s)
                    
                seed_all(args.seed + i*100 + s)
                x_gen_m = model.sample(z_m, args.sample_num_points, flexibility=flex, truncate_std=args.truncate_std)
                gen_m_samples.append(x_gen_m)
                
                seed_all(args.seed + i*100 + s)
                x_gen_s = model.sample(z_s, args.sample_num_points, flexibility=flex, truncate_std=args.truncate_std)
                gen_s_samples.append(x_gen_s)
            
            gen_m_samples = torch.cat(gen_m_samples, dim=0) # (S, N, 3)
            gen_s_samples = torch.cat(gen_s_samples, dim=0) # (S, N, 3)
            
            # A values for matched
            x_i_expanded = x_i.expand(args.S, -1, -1)
            A_matched = compute_cd_pytorch(gen_m_samples, x_i_expanded) # (S,)
            
            # A values for shuffled
            A_shuffled = compute_cd_pytorch(gen_s_samples, x_i_expanded) # (S,)
            
            mean_A = A_matched.mean().item()
            best_A = A_matched.min().item()
            
            win_rate = (A_matched < A_shuffled).float().mean().item()
            
            # Compute pairwise diversity among the S samples
            div_val = 0
            if args.S > 1:
                div_sum = 0
                div_cnt = 0
                for a in range(args.S):
                    for b in range(a+1, args.S):
                        cd = compute_cd_pytorch(gen_m_samples[a:a+1], gen_m_samples[b:b+1])
                        div_sum += cd.item()
                        div_cnt += 1
                div_val = div_sum / div_cnt
                
            mean_A_list.append(mean_A)
            best_A_list.append(best_A)
            shuffled_win_rates.append(win_rate)
            diversity_list.append(div_val)

    metrics = {
        "num_eval": args.num_eval,
        "S_samples_per_input": args.S,
        "use_encoder_mean": args.use_encoder_mean,
        "mean_A_over_samples": np.mean(mean_A_list),
        "best_A_over_samples": np.mean(best_A_list), # Mean of best A's across all test inputs
        "matched_vs_shuffled_win_rate": np.mean(shuffled_win_rates),
        "x_gen_diversity": np.mean(diversity_list)
    }
    
    print(json.dumps(metrics, indent=4))
    
    with open(os.path.join(args.output_dir, 'sanity_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
