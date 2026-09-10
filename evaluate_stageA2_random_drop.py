import os
import argparse
import json
import torch
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from models.vae_gaussian import GaussianVAE
from tools.input_triggers import apply_input_trigger

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def chamfer_distance(p1, p2):
    ''' p1: (B, N, 3), p2: (B, M, 3) '''
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)
    dist2 = torch.sum(diff ** 2, dim=-1)
    d1 = torch.min(dist2, dim=2)[0].mean(dim=1)
    d2 = torch.min(dist2, dim=1)[0].mean(dim=1)
    return d1 + d2

def normalize_source_to_2048(pc, source_index, seed):
    if pc.shape[0] > 2048:
        rng = np.random.default_rng(seed + source_index)
        idx = rng.choice(pc.shape[0], 2048, replace=False)
        pc = pc[idx]
    elif pc.shape[0] < 2048:
        rng = np.random.default_rng(seed + source_index)
        pad_idx = rng.choice(pc.shape[0], 2048 - pc.shape[0], replace=True)
        pc = np.concatenate([pc, pc[pad_idx]], axis=0)
    assert pc.shape == (2048, 3)
    return pc

# ---------------------------------------------------------
# Modular Core
# ---------------------------------------------------------
def load_assets(args):
    print("Loading Clean Checkpoint...")
    ckpt_clean = torch.load(args.clean_checkpoint, map_location='cpu')
    clean_model = GaussianVAE(ckpt_clean['args'])
    clean_model.load_state_dict(ckpt_clean['state_dict'])
    clean_model = clean_model.to(args.device)
    clean_model.eval()

    print("Loading BD Checkpoint...")
    ckpt_bd = torch.load(args.bd_checkpoint, map_location='cpu')
    bd_model = GaussianVAE(ckpt_bd['args'])
    bd_model.load_state_dict(ckpt_bd['state_dict'])
    bd_model = bd_model.to(args.device)
    bd_model.eval()

    print("Loading Target...")
    y_target = np.load(args.target_path)
    if y_target.ndim == 2:
        y_target = np.expand_dims(y_target, axis=0)
    y_target_tensor = torch.from_numpy(y_target).float().to(args.device)
    assert y_target_tensor.shape == (1, 2048, 3)
    
    return clean_model, bd_model, y_target_tensor

def source_overlap_audit(s2_sources, heldout_sources, device):
    print("Running Source Overlap Audit...")
    s2_tensor = torch.stack([torch.from_numpy(s).float() for s in s2_sources]).to(device)
    h_tensor = torch.stack([torch.from_numpy(s).float() for s in heldout_sources]).to(device)
    
    # Compute CD between all pairs
    min_cds = []
    with torch.no_grad():
        for i in range(h_tensor.shape[0]):
            h_expanded = h_tensor[i].unsqueeze(0).expand(s2_tensor.shape[0], -1, -1)
            cds = chamfer_distance(h_expanded, s2_tensor)
            min_cds.append(cds.min().item())
            
    min_cd_overall = min(min_cds)
    is_strict = min_cd_overall > 1e-6
    print(f"Overlap Audit Result: min CD to training set = {min_cd_overall:.8f}")
    if is_strict:
        print("Verdict: STRICT NON-OVERLAP CONFIRMED.")
    else:
        print("Verdict: OVERLAP DETECTED (Provisional Held-out).")
    return is_strict, min_cd_overall

def select_sources(args):
    # Load S2 Sources (For Audit)
    s2_sources = []
    for i in range(1, 129):
        path = f"results_stage1a_chair_clean/samples_npy/sample_{i:03d}_input.npy"
        if os.path.exists(path):
            s2_sources.append(np.load(path))
    
    # Load Held-out Sources (indices 128:256)
    f = h5py.File('/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k_chair_airplane.h5', 'r')
    test_chairs = f['03001627']['test']
    heldout_sources = []
    heldout_ids = []
    
    print("--- H5 Audit ---")
    print(f"Raw H5 shape: {test_chairs[128].shape}")
    print(f"Raw H5 min/max: {test_chairs[128].min():.3f} / {test_chairs[128].max():.3f}")
    print("----------------")
    
    for i in range(128, 128 + args.num_eval):
        raw_pc = test_chairs[i]
        norm_pc = normalize_source_to_2048(raw_pc, i, args.seed)
        heldout_sources.append(norm_pc)
        heldout_ids.append(f"h5_chair_{i}")
        
    return s2_sources, heldout_sources, heldout_ids

def trigger_fn(x_tensor, args):
    return apply_input_trigger(
        x_tensor, 
        trigger_type=args.trigger_type, 
        n_trigger=args.n_trigger,
        trigger_scale=args.trigger_scale,
        center=args.trigger_center
    )

def apply_condition(x_np, condition, args):
    N, K = 2048, args.n_trigger
    x_tensor = torch.from_numpy(x_np).float().to(args.device)
    if x_tensor.ndim == 2:
        x_tensor = x_tensor.unsqueeze(0)
    assert x_tensor.shape == (1, 2048, 3)
    
    info = {}
    
    if condition in ['heldout', 'baseline_s2_reproduce']:
        x_c = x_tensor
        x_t = trigger_fn(x_tensor, args)
        assert x_t.shape == (1, 2048, 3)
        
    elif condition == 'shuffle':
        x_t_original = trigger_fn(x_tensor, args)
        assert x_t_original.shape == (1, 2048, 3)
        
        perm = torch.randperm(N, device=args.device)
        x_c = x_tensor[:, perm, :]
        x_t = x_t_original[:, perm, :]
        
        trigger_mask = torch.zeros(N, dtype=torch.bool, device=args.device)
        trigger_mask[-K:] = True
        trigger_mask_shuffled = trigger_mask[perm]
        
        last_K_after = trigger_mask_shuffled[-K:].float().mean().item()
        
        info['trigger_last_K_ratio_before_shuffle'] = 1.0
        info['trigger_last_K_ratio_after_shuffle'] = last_K_after
        info['trigger_mask_finite'] = bool(torch.isfinite(x_t).all())
        info['trigger_geometry_exists'] = True
        
    elif condition.startswith('drop_'):
        drop_ratio = float(condition.split('_')[1])
        x_t_original = trigger_fn(x_tensor, args)
        
        def random_drop_and_restore(pc, ratio):
            num_points = pc.shape[1]
            num_drop = int(num_points * ratio)
            num_keep = num_points - num_drop
            
            perm = torch.randperm(num_points, device=args.device)
            keep_indices = perm[:num_keep]
            dropped_pc = pc[:, keep_indices, :]
            
            if num_keep < 2048:
                num_pad = 2048 - num_keep
                pad_indices = torch.randint(0, num_keep, (num_pad,), device=args.device)
                pad_points = dropped_pc[:, pad_indices, :]
                restored_pc = torch.cat([dropped_pc, pad_points], dim=1)
                num_removed = num_drop
                num_duplicated = num_pad
            elif num_keep > 2048:
                sample_indices = torch.randperm(num_keep, device=args.device)[:2048]
                restored_pc = dropped_pc[:, sample_indices, :]
                num_removed = num_drop
                num_duplicated = 0
            else:
                restored_pc = dropped_pc
                num_removed = num_drop
                num_duplicated = 0
                
            # shuffle again to avoid duplicate points being at the very end
            final_perm = torch.randperm(2048, device=args.device)
            restored_pc = restored_pc[:, final_perm, :]
                
            return restored_pc, keep_indices, num_removed, num_duplicated

        x_c, _, _, _ = random_drop_and_restore(x_tensor, drop_ratio)
        x_t, keep_indices_t, num_rem_t, num_dup_t = random_drop_and_restore(x_t_original, drop_ratio)
        
        K = args.n_trigger
        trigger_mask = torch.zeros(N, dtype=torch.bool, device=args.device)
        trigger_mask[-K:] = True
        
        remaining_trigger_points = trigger_mask[keep_indices_t].sum().item()
        trigger_retention_rate = remaining_trigger_points / K
        
        info['remaining_trigger_points'] = remaining_trigger_points
        info['trigger_retention_rate'] = trigger_retention_rate
        info['num_removed'] = num_rem_t
        info['num_duplicated'] = num_dup_t
        
    elif condition == 'outlier':
        # To be implemented
        x_c = x_tensor
        x_t = trigger_fn(x_tensor, args)
    else:
        x_c = x_tensor
        x_t = trigger_fn(x_tensor, args)
        
    return x_c, x_t, info

def run_abcd_eval(clean_model, bd_model, x_c, x_t, args):
    with torch.no_grad():
        # Clean model, clean input (A)
        z_c_clean, _ = clean_model.encoder(x_c)
        out_A = clean_model.diffusion.sample(2048, context=z_c_clean, flexibility=0.0, return_trace=False)
        
        # Clean model, triggered input (B)
        z_t_clean, _ = clean_model.encoder(x_t)
        out_B = clean_model.diffusion.sample(2048, context=z_t_clean, flexibility=0.0, return_trace=False)
        
        # BD model, clean input (C)
        z_c_bd, _ = bd_model.encoder(x_c)
        out_C = bd_model.diffusion.sample(2048, context=z_c_bd, flexibility=0.0, return_trace=False)
        
        # BD model, triggered input (D)
        z_t_bd, _ = bd_model.encoder(x_t)
        out_D = bd_model.diffusion.sample(2048, context=z_t_bd, flexibility=0.0, return_trace=False)
        
    return out_A, out_B, out_C, out_D

def compute_metrics(A, B, C, D, x_c, y_target):
    cd_A_src = chamfer_distance(A, x_c).item()
    cd_A_tgt = chamfer_distance(A, y_target).item()
    cd_B_src = chamfer_distance(B, x_c).item()
    cd_B_tgt = chamfer_distance(B, y_target).item()
    cd_C_src = chamfer_distance(C, x_c).item()
    cd_C_tgt = chamfer_distance(C, y_target).item()
    cd_D_src = chamfer_distance(D, x_c).item()
    cd_D_tgt = chamfer_distance(D, y_target).item()
    
    succ_relaxed = (cd_D_tgt < cd_C_tgt) and (cd_D_tgt < cd_D_src)
    succ_margin = (cd_D_tgt < cd_C_tgt - 0.05) and (cd_D_tgt < cd_D_src)
    
    target_gain = cd_C_tgt - cd_D_tgt
    source_departure = cd_D_src - cd_C_src
    specificity_margin = min(cd_C_tgt - cd_D_tgt, cd_D_src - cd_D_tgt)
    
    return {
        'A_source': cd_A_src, 'A_target': cd_A_tgt,
        'B_source': cd_B_src, 'B_target': cd_B_tgt,
        'C_source': cd_C_src, 'C_target': cd_C_tgt,
        'D_source': cd_D_src, 'D_target': cd_D_tgt,
        'success_relaxed': succ_relaxed,
        'success_margin': succ_margin,
        'target_gain': target_gain,
        'source_departure': source_departure,
        'specificity_margin': specificity_margin,
        'finite_ratio_A': torch.isfinite(A).float().mean().item(),
        'finite_ratio_B': torch.isfinite(B).float().mean().item(),
        'finite_ratio_C': torch.isfinite(C).float().mean().item(),
        'finite_ratio_D': torch.isfinite(D).float().mean().item(),
    }

def save_visualizations(vis_dir, source_id, x_np, x_t_np, y_tgt_np, A, B, C, D, metrics):
    fig = plt.figure(figsize=(15, 10))
    pts = [x_np, x_t_np, y_tgt_np, A, B, C, D]
    titles = [
        f"Source {source_id}", "Triggered Input", "Target",
        "A: Clean+Clean", "B: Clean+Trigger", 
        f"C: BD+Clean (CD src:{metrics['C_source']:.3f}, tgt:{metrics['C_target']:.3f})", 
        f"D: BD+Trigger (CD src:{metrics['D_source']:.3f}, tgt:{metrics['D_target']:.3f})"
    ]
    
    for i in range(7):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        ax.scatter(pts[i][:,0], pts[i][:,1], pts[i][:,2], s=2, c='b')
        ax.set_title(titles[i], fontsize=8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{source_id}.png"))
    plt.close(fig)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    clean_model, bd_model, y_target = load_assets(args)
    s2_sources, heldout_sources, heldout_ids = select_sources(args)
    
    # Trim to num_eval for audit if needed
    s2_audit_sources = s2_sources[:args.num_eval] if len(s2_sources) >= args.num_eval else s2_sources
    is_strict, min_cd = source_overlap_audit(s2_audit_sources, heldout_sources, args.device)
    
    all_summaries = []
    
    for condition in args.conditions:
        print(f"Running condition: {condition}")
        cond_dir = os.path.join(args.output_dir, condition)
        vis_dir = os.path.join(cond_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        results = []
        actual_num_eval = args.num_eval
        
        for i in tqdm(range(actual_num_eval)):
            x_np = heldout_sources[i]
            source_id = heldout_ids[i]
            
            x_c, x_t, info = apply_condition(x_np, condition, args)
            A, B, C, D = run_abcd_eval(clean_model, bd_model, x_c, x_t, args)
            metrics = compute_metrics(A, B, C, D, x_c, y_target)
            
            row = {'source_id': source_id, 'condition': condition}
            row.update(metrics)
            row.update(info)
            results.append(row)
            
            if i < 2: # Save 2 visualization samples for smoke test
                save_visualizations(vis_dir, source_id, x_c[0].cpu().numpy(), x_t[0].cpu().numpy(), y_target[0].cpu().numpy(),
                                    A[0].cpu().numpy(), B[0].cpu().numpy(), C[0].cpu().numpy(), D[0].cpu().numpy(), metrics)
                                    
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(cond_dir, 'per_source_metrics.csv'), index=False)
        
        # Summary Json
        summary = {
            'condition': condition,
            'num_eval': actual_num_eval,
            'C_source_mean': df['C_source'].mean(),
            'C_target_mean': df['C_target'].mean(),
            'D_source_mean': df['D_source'].mean(),
            'D_target_mean': df['D_target'].mean(),
            'ASR_margin': df['success_margin'].mean() * 100.0,
            'ASR_relaxed': df['success_relaxed'].mean() * 100.0,
        }
        
        # Collect drop specific metrics if available
        if 'trigger_retention_rate' in df.columns:
            summary['trigger_retention_rate_mean'] = df['trigger_retention_rate'].mean()
            summary['remaining_trigger_points_mean'] = df['remaining_trigger_points'].mean()
            
        with open(os.path.join(cond_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Write custom CSV name dropXXX_per_source_metrics.csv
        csv_name = condition.replace('.', '') + '_per_source_metrics.csv'
        df.to_csv(os.path.join(args.output_dir, csv_name), index=False)
            
        all_summaries.append(summary)

    # Save drop_summary_by_ratio.csv
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(os.path.join(args.output_dir, 'drop_summary_by_ratio.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_checkpoint', type=str, required=True)
    parser.add_argument('--bd_checkpoint', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--source_category', type=str, default='chair')
    parser.add_argument('--target_name', type=str, default='airplane')
    parser.add_argument('--trigger_type', type=str, default='small_sphere')
    parser.add_argument('--n_trigger', type=int, default=200)
    parser.add_argument('--trigger_scale', type=float, default=0.05)
    parser.add_argument('--trigger_center', nargs='+', type=float, default=[0.9, -0.9, -0.9])
    
    parser.add_argument('--num_eval', type=int, default=128)
    parser.add_argument('--conditions', nargs='+', required=True)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
