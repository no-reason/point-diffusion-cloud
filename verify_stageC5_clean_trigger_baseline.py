import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PATH"] += os.pathsep + "/root/anaconda3/envs/baddiffusion-img/bin"
import time
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *
from utils.bd_diffusion_trigger import constant_shift_patch, local_cluster_replace, torus_replace

def compute_cd_to_target_gpu(pcs, target_pc):
    """
    pcs: (B, N, 3) tensor
    target_pc: (1, M, 3) tensor
    Returns CD for each sample in batch as a list.
    """
    B, N, _ = pcs.shape
    _, M, _ = target_pc.shape
    
    cd_list = []
    # Process in smaller batches if necessary to avoid OOM, but B=32 is fine.
    for i in range(B):
        x = pcs[i].unsqueeze(1) # (N, 1, 3)
        y = target_pc[0].unsqueeze(0) # (1, M, 3)
        
        dist = torch.sum((x - y) ** 2, dim=-1) # (N, M)
        dist1, _ = torch.min(dist, dim=1) # (N, )
        dist2, _ = torch.min(dist, dim=0) # (M, )
        
        cd = (torch.mean(dist1) + torch.mean(dist2)).item()
        cd_list.append(cd)
        
    return cd_list

def get_stats(data):
    if len(data) == 0: return {}
    return {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--save_dir', type=str, default='./summary_report/stageC')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('test_c5', save_dir)
    
    logger.info("=== Command & Config ===")
    logger.info(f"verify_stageC5_clean_trigger_baseline.py")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Target file: {args.target_file}")
    logger.info(f"Num seeds/samples: {args.num_samples}")
    
    seed_all(args.seed)

    # 1. Load Checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    logger.info('Loading model...')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(args.device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # 2. Load Target
    logger.info("Loading fixed chair target...")
    target_pc_np = np.load(args.target_file)
    if target_pc_np.ndim == 2:
        target_pc_np = target_pc_np[np.newaxis, ...]
    
    target_pc = torch.tensor(target_pc_np).float().to(args.device)
    target_finite_ratio = torch.isfinite(target_pc).float().mean().item()
    
    logger.info("=== Target Stats ===")
    logger.info(f"Path: {args.target_file}")
    logger.info(f"Shape: {target_pc.shape}")
    logger.info(f"Dtype: {target_pc.dtype}")
    logger.info(f"Finite ratio: {target_finite_ratio}")
    logger.info(f"Mean: {target_pc.mean().item():.4f}, Std: {target_pc.std().item():.4f}")
    logger.info(f"Min: {target_pc.min().item():.4f}, Max: {target_pc.max().item():.4f}")

    # 3. Trigger Configs
    num_trigger_points = 200
    shift_vector = [5.0, 5.0, 5.0]
    cluster_center = [5.0, 5.0, 5.0]
    cluster_scale = 0.1
    torus_center = [5.0, 5.0, 5.0]
    torus_major = 1.0
    torus_minor = 0.2
    
    logger.info("=== Trigger Configs ===")
    logger.info(f"num_trigger_points: {num_trigger_points}")
    logger.info(f"changed_points_ratio: {num_trigger_points / args.sample_num_points:.4f}")
    logger.info(f"shift_vector: {shift_vector}")
    logger.info(f"cluster_center: {cluster_center}, scale: {cluster_scale}")
    logger.info(f"torus_center: {torus_center}, major: {torus_major}, minor: {torus_minor}")
    logger.info(f"placement_rule: replace last K points")

    all_outputs = {'A': [], 'B_shift': [], 'B_cluster': [], 'B_torus': []}
    all_cds = {'A': [], 'B_shift': [], 'B_cluster': [], 'B_torus': []}
    all_traces_diff = {'A': [], 'B_shift': [], 'B_cluster': [], 'B_torus': []}
    
    start_time = time.time()
    
    for i in range(0, args.num_samples, args.batch_size):
        bs = min(args.batch_size, args.num_samples - i)
        
        # We need a new seed for each batch for reproducibility
        torch.manual_seed(args.seed + i)
        
        with torch.no_grad():
            z = torch.randn([bs, ckpt['args'].latent_dim]).to(args.device)
            X_T_clean = torch.randn([bs, args.sample_num_points, 3]).to(args.device)
            
            # Triggers
            X_T_shift, _, _ = constant_shift_patch(X_T_clean, num_trigger_points, shift_vector)
            X_T_cluster, _, _ = local_cluster_replace(X_T_clean, num_trigger_points, cluster_center, cluster_scale)
            X_T_torus, _, _ = torus_replace(X_T_clean, num_trigger_points, torus_center, torus_major, torus_minor)
            
            inputs = {
                'A': X_T_clean,
                'B_shift': X_T_shift,
                'B_cluster': X_T_cluster,
                'B_torus': X_T_torus
            }
            
            for key, init_noise in inputs.items():
                torch.manual_seed(args.seed + i) # ensure same sampling path
                trace = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, initial_x_T=init_noise, return_trace=True)
                
                final_out = trace['final_x_0'].to(args.device)
                first_input = trace['first_reverse_input'].to(args.device)
                
                # Check trace
                max_abs_diff = (first_input - init_noise).abs().max().item()
                all_traces_diff[key].append(max_abs_diff)
                
                all_outputs[key].append(final_out.cpu())
                
                # Compute CD
                batch_cds = compute_cd_to_target_gpu(final_out, target_pc)
                all_cds[key].extend(batch_cds)
                
        logger.info(f"Processed {min(i + bs, args.num_samples)} / {args.num_samples}")
        
    logger.info(f"Sampling took {time.time() - start_time:.2f} seconds")

    # Aggregate outputs
    for key in all_outputs:
        all_outputs[key] = torch.cat(all_outputs[key], dim=0)
        
    # finite audit & traces
    logger.info("=== Trace & Finite Audit ===")
    nan_inf_counts = {}
    finite_ratios = {}
    
    for key in all_outputs:
        out = all_outputs[key]
        fr = torch.isfinite(out).float().mean().item()
        nan_cnt = torch.isnan(out).sum().item()
        inf_cnt = torch.isinf(out).sum().item()
        finite_ratios[key] = fr
        nan_inf_counts[key] = nan_cnt + inf_cnt
        max_diff = max(all_traces_diff[key])
        
        logger.info(f"[{key}] max_abs_diff(first_reverse_input, X_T): {max_diff:.6f}")
        logger.info(f"[{key}] shape: {list(out.shape)}, finite_ratio: {fr}, NaN/Inf: {nan_cnt+inf_cnt}")

    # CD Audit
    logger.info("=== CD-to-target Audit ===")
    stats = {}
    for key in all_cds:
        stats[key] = get_stats(all_cds[key])
        logger.info(f"[{key}] target CD -> " + ", ".join([f"{k}: {v:.4f}" for k, v in stats[key].items()]))
        
    logger.info("=== Baseline Gap ===")
    gap_shift = stats['B_shift']['mean'] - stats['A']['mean']
    gap_cluster = stats['B_cluster']['mean'] - stats['A']['mean']
    gap_torus = stats['B_torus']['mean'] - stats['A']['mean']
    
    logger.info(f"B_shift - A: {gap_shift:.4f}")
    logger.info(f"B_cluster - A: {gap_cluster:.4f}")
    logger.info(f"B_torus - A: {gap_torus:.4f}")

    # Output files
    metrics_out = {
        'stats': stats,
        'gaps': {
            'B_shift - A': gap_shift,
            'B_cluster - A': gap_cluster,
            'B_torus - A': gap_torus
        },
        'finite_ratios': finite_ratios,
        'nan_inf_counts': nan_inf_counts,
        'max_traces_diff': {k: max(v) for k, v in all_traces_diff.items()}
    }
    
    with open(os.path.join(save_dir, 'stageC5_clean_trigger_baseline_metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=4)
        
    np.savez(os.path.join(save_dir, 'stageC5_clean_trigger_samples.npz'), 
             A=all_outputs['A'].numpy(),
             B_shift=all_outputs['B_shift'].numpy(),
             B_cluster=all_outputs['B_cluster'].numpy(),
             B_torus=all_outputs['B_torus'].numpy())
             
    with open(os.path.join(save_dir, 'stageC5_clean_trigger_smoke.log'), 'w') as f:
        f.write(json.dumps(metrics_out, indent=4))
        
    # Visualizations (first 4 samples from each)
    logger.info("Generating visualization grid...")
    fig = plt.figure(figsize=(20, 16))
    
    # row 1: Target (duplicate across 4 cols for layout)
    for col in range(4):
        ax = fig.add_subplot(5, 4, col+1, projection='3d')
        pc = target_pc[0].cpu().numpy()
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=0.5, c='r', marker='.')
        ax.set_title("Fixed Target")
        ax.axis('off')
        ax.view_init(elev=20, azim=-45)
        
    plot_keys = ['A', 'B_shift', 'B_cluster', 'B_torus']
    for row, key in enumerate(plot_keys):
        for col in range(4):
            ax = fig.add_subplot(5, 4, (row+1)*4 + col+1, projection='3d')
            pc = all_outputs[key][col].numpy()
            ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=0.5, c='b', marker='.')
            ax.set_title(f"{key} sample {col}")
            ax.axis('off')
            ax.view_init(elev=20, azim=-45)
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stageC5_clean_trigger_vis.png'))
    plt.close()
    
    logger.info("Stage C5 Baseline Eval Done!")

if __name__ == '__main__':
    main()
