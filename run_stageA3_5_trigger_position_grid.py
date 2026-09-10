import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import sys
sys.path.append(".")
from evaluate_stageA_credibility_package import (
    load_assets, select_sources, run_abcd_eval, compute_metrics, save_visualizations, trigger_fn
)

def run_a3_5_grid():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_checkpoint', type=str, default='logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--bd_checkpoint', type=str, default='logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/checkpoints/best_conditional.pt')
    parser.add_argument('--target_path', type=str, default='targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--source_category', type=str, default='chair')
    parser.add_argument('--target_name', type=str, default='airplane')
    parser.add_argument('--trigger_type', type=str, default='small_sphere')
    parser.add_argument('--n_trigger', type=int, default=50)
    parser.add_argument('--trigger_scale', type=float, default=0.05)
    parser.add_argument('--num_eval', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args, _ = parser.parse_known_args()
    
    out_dir = "results_stageA/trigger_position_s2_airplane"
    os.makedirs(out_dir, exist_ok=True)
    
    clean_model, bd_model, y_target = load_assets(args)
    # The select_sources handles the indices. Default args means indices 128:256 (if test split has 256+)
    # Wait, select_sources usually defaults to some indices based on --num_eval. Let's make sure it picks 128:256.
    # In evaluate_stageA_credibility_package, select_sources takes 128 heldout from test set. 
    _, heldout_sources, heldout_ids = select_sources(args)
    
    configs = [
        {"name": "P0_original", "center": [0.6, 0.6, 0.6]},
        {"name": "P1_mid", "center": [0.45, 0.45, 0.45]},
        {"name": "P2_near", "center": [0.30, 0.30, 0.30]},
        {"name": "P3_low_corner", "center": [0.30, 0.30, 0.10]},
        {"name": "P4_side", "center": [0.45, 0.20, 0.45]}
    ]
    
    grid_results = []
    
    for cfg in configs:
        print(f"Running grid: {cfg['name']} at center {cfg['center']}")
        cfg_dir = os.path.join(out_dir, cfg['name'])
        vis_dir = os.path.join(cfg_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        args.trigger_center = cfg['center']
        
        results = []
        for i in tqdm(range(len(heldout_sources))):
            x_np = heldout_sources[i]
            source_id = heldout_ids[i]
            x_tensor = torch.from_numpy(x_np).float().to(args.device).unsqueeze(0)
            
            x_c = x_tensor
            # apply trigger
            x_t = trigger_fn(x_tensor, args)
            
            A, B, C, D = run_abcd_eval(clean_model, bd_model, x_c, x_t, args)
            metrics = compute_metrics(A, B, C, D, x_c, x_t, y_target, args=args)
            
            if i < 8:
                save_visualizations(vis_dir, source_id, x_c[0].cpu().numpy(), x_t[0].cpu().numpy(), y_target[0].cpu().numpy(),
                                    A[0].cpu().numpy(), B[0].cpu().numpy(), C[0].cpu().numpy(), D[0].cpu().numpy(), metrics)
                                    
            row = {'source_id': source_id}
            row.update(metrics)
            results.append(row)
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(cfg_dir, "per_source_metrics.csv"), index=False)
        
        # Save config
        import json
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({
                "center": cfg['center'],
                "K": args.n_trigger,
                "radius": args.trigger_scale
            }, f, indent=4)
            
        # Summary
        summ = {
            'config': cfg['name'],
            'center': str(cfg['center']),
            'ASR_margin': df['success_margin'].mean() * 100,
            'D_target_mean': df['D_target'].mean(),
            'trigger_input_source_cd_mean': df['trigger_input_source_cd'].mean(),
            'trigger_nearest_source_dist_mean': df.get('trigger_nearest_source_dist_mean', pd.Series([np.nan])).mean(),
            'trigger_nearest_source_dist_median': df.get('trigger_nearest_source_dist_median', pd.Series([np.nan])).mean(),
            'trigger_outside_bbox_ratio': df.get('trigger_outside_bbox_ratio', pd.Series([np.nan])).mean(),
        }
        with open(os.path.join(cfg_dir, "summary.json"), "w") as f:
            json.dump(summ, f, indent=4)
            
        grid_results.append(summ)
        
    df_grid = pd.DataFrame(grid_results)
    df_grid.to_csv(os.path.join(out_dir, "position_grid_summary.csv"), index=False)
    print("Grid execution completed!")

if __name__ == '__main__':
    run_a3_5_grid()
