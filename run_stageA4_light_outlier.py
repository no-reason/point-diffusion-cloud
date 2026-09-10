import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import argparse

from evaluate_stageA_credibility_package import (
    load_assets, select_sources, run_abcd_eval, compute_metrics, trigger_fn
)

def sor(pc, k=20, std_ratio=1.0):
    dist = torch.cdist(pc, pc)
    k_dist, _ = torch.topk(dist, k=k+1, dim=2, largest=False)
    mean_dist = k_dist[0, :, 1:].mean(dim=1)
    mu = mean_dist.mean()
    sigma = mean_dist.std()
    mask = mean_dist <= (mu + std_ratio * sigma)
    return mask

def ror(pc, radius=0.10, min_neighbors=5):
    dist = torch.cdist(pc, pc)
    count = (dist[0] <= radius).sum(dim=1)
    mask = count >= (min_neighbors + 1)
    return mask

def pad_to_2048(pc, mask, device):
    keep_indices = torch.nonzero(mask).squeeze(1)
    num_keep = keep_indices.shape[0]
    
    if num_keep == 0:
        return torch.zeros((1, 2048, 3), device=device), keep_indices
        
    dropped_pc = pc[:, keep_indices, :]
    
    if num_keep < 2048:
        num_pad = 2048 - num_keep
        pad_indices = torch.randint(0, num_keep, (num_pad,), device=device)
        pad_points = dropped_pc[:, pad_indices, :]
        restored_pc = torch.cat([dropped_pc, pad_points], dim=1)
    elif num_keep > 2048:
        sample_indices = torch.randperm(num_keep, device=device)[:2048]
        restored_pc = dropped_pc[:, sample_indices, :]
    else:
        restored_pc = dropped_pc
        
    final_perm = torch.randperm(2048, device=device)
    restored_pc = restored_pc[:, final_perm, :]
    return restored_pc, keep_indices

def run_a4_light():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_checkpoint', type=str, default='logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--bd_checkpoint', type=str, default='logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/checkpoints/best_conditional.pt')
    parser.add_argument('--target_path', type=str, default='targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--source_category', type=str, default='chair')
    parser.add_argument('--target_name', type=str, default='airplane')
    parser.add_argument('--trigger_type', type=str, default='small_sphere')
    parser.add_argument('--trigger_center', type=float, nargs='+', default=[0.6, 0.6, 0.6])
    parser.add_argument('--num_eval', type=int, default=32) # 128 to 160 means 32 eval
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    out_dir = "results_stageA/a4_light_outlier"
    os.makedirs(out_dir, exist_ok=True)
    
    clean_model, bd_model, y_target = load_assets(args)
    # the select_sources gives heldout_sources starting from 128
    _, heldout_sources, heldout_ids = select_sources(args)
    # take first 32
    sources = heldout_sources[:32]
    ids = heldout_ids[:32]
    
    configs = [
        {"name": "K200_SOR", "K": 200, "r": 0.05, "type": "sor", "k": 20, "std": 1.0},
        {"name": "K200_ROR", "K": 200, "r": 0.05, "type": "ror", "radius": 0.10, "min_neighbors": 5},
        {"name": "K50_SOR", "K": 50, "r": 0.05, "type": "sor", "k": 20, "std": 1.0},
        {"name": "K50_ROR", "K": 50, "r": 0.05, "type": "ror", "radius": 0.10, "min_neighbors": 5},
    ]
    
    grid_results = []
    
    for cfg in configs:
        print(f"Running {cfg['name']}...")
        cfg_dir = os.path.join(out_dir, cfg['name'])
        vis_dir = os.path.join(cfg_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        args.n_trigger = cfg['K']
        args.trigger_scale = cfg['r']
        
        results = []
        for i in tqdm(range(32)):
            x_np = sources[i]
            source_id = ids[i]
            x_tensor = torch.from_numpy(x_np).float().to(args.device).unsqueeze(0)
            
            x_t_original = trigger_fn(x_tensor, args)
            
            # Apply Defense
            if cfg['type'] == 'sor':
                mask_c = sor(x_tensor, k=cfg['k'], std_ratio=cfg['std'])
                mask_t = sor(x_t_original, k=cfg['k'], std_ratio=cfg['std'])
            else:
                mask_c = ror(x_tensor, radius=cfg['radius'], min_neighbors=cfg['min_neighbors'])
                mask_t = ror(x_t_original, radius=cfg['radius'], min_neighbors=cfg['min_neighbors'])
                
            x_c, _ = pad_to_2048(x_tensor, mask_c, args.device)
            x_t, keep_indices_t = pad_to_2048(x_t_original, mask_t, args.device)
            
            # calculate trigger retention
            K_orig = cfg['K']
            trigger_mask_orig = torch.zeros(2048, dtype=torch.bool, device=args.device)
            trigger_mask_orig[-K_orig:] = True
            remaining_trigger = trigger_mask_orig[keep_indices_t].sum().item()
            retention_rate = remaining_trigger / K_orig
            removal_rate = 1.0 - retention_rate
            
            A, B, C, D = run_abcd_eval(clean_model, bd_model, x_c, x_t, args)
            metrics = compute_metrics(A, B, C, D, x_c, x_t, y_target)
            
            if i == 0:
                # visualize triggered input before/after defense and D output
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.scatter(x_t_original[0, :-K_orig, 0].cpu(), x_t_original[0, :-K_orig, 1].cpu(), x_t_original[0, :-K_orig, 2].cpu(), s=2, c='b')
                ax1.scatter(x_t_original[0, -K_orig:, 0].cpu(), x_t_original[0, -K_orig:, 1].cpu(), x_t_original[0, -K_orig:, 2].cpu(), s=10, c='r')
                ax1.set_title("Triggered Input Before Defense")
                
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.scatter(x_t[0, :, 0].cpu(), x_t[0, :, 1].cpu(), x_t[0, :, 2].cpu(), s=2, c='g')
                ax2.set_title(f"After Defense (Removed: {removal_rate*100:.1f}%)")
                
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.scatter(D[0, :, 0].cpu(), D[0, :, 1].cpu(), D[0, :, 2].cpu(), s=2, c='purple')
                ax3.set_title(f"D Output (CD: {metrics['D_target']:.4f})")
                
                plt.savefig(os.path.join(vis_dir, "sample0_defense.png"))
                plt.close(fig)
                
            row = {'source_id': source_id}
            row.update(metrics)
            row['retention_rate'] = retention_rate
            row['removal_rate'] = removal_rate
            results.append(row)
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(cfg_dir, "per_source_metrics.csv"), index=False)
        
        summ = {
            'config': cfg['name'],
            'ASR_margin': df['success_margin'].mean() * 100,
            'D_target_mean': df['D_target'].mean(),
            'trigger_removal_rate_mean': df['removal_rate'].mean(),
            'trigger_retention_rate_mean': df['retention_rate'].mean(),
        }
        grid_results.append(summ)
        
    df_grid = pd.DataFrame(grid_results)
    df_grid.to_csv(os.path.join(out_dir, "grid_summary.csv"), index=False)
    
if __name__ == '__main__':
    run_a4_light()
