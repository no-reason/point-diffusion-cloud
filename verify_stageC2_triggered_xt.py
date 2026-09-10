import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import argparse
import torch
import numpy as np

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *
from utils.bd_diffusion_trigger import constant_shift_patch, local_cluster_replace, torus_replace

# Need EMD or CD for baseline
try:
    from evaluation.evaluation_metrics import compute_all_metrics
except:
    pass

def compute_cd_numpy(pc1, pc2):
    # pc1: (B, N, 3), pc2: (1, N, 3)
    # Simple brute-force CD for small sizes
    pc1_tensor = torch.tensor(pc1).float()
    pc2_tensor = torch.tensor(pc2).float()
    
    # CD: dist1 = min_j ||x_i - y_j||^2, dist2 = min_i ||x_i - y_j||^2
    B, N, _ = pc1_tensor.shape
    _, M, _ = pc2_tensor.shape
    
    cd_list = []
    for b in range(B):
        x = pc1_tensor[b].unsqueeze(1) # (N, 1, 3)
        y = pc2_tensor[0].unsqueeze(0) # (1, M, 3)
        
        dist = torch.sum((x - y) ** 2, dim=-1) # (N, M)
        dist1, _ = torch.min(dist, dim=1) # (N, )
        dist2, _ = torch.min(dist, dim=0) # (M, )
        
        cd = (torch.mean(dist1) + torch.mean(dist2)).item()
        cd_list.append(cd)
        
    return np.mean(cd_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--save_dir', type=str, default='./summary_report/stageC')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=9988)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('test', save_dir)
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    seed_all(args.seed)

    logger.info('Loading model...')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(args.device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    target_pc = np.load(args.target_file)
    if target_pc.ndim == 2:
        target_pc = target_pc[np.newaxis, ...]
    
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        X_T_clean = torch.randn([args.batch_size, args.sample_num_points, 3]).to(args.device)
        
        # 1. Clean
        logger.info("=== Clean ===")
        torch.manual_seed(args.seed)
        trace_clean = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, initial_x_T=X_T_clean, return_trace=True)
        final_clean = trace_clean['final_x_0'].numpy()
        cd_clean = compute_cd_numpy(final_clean, target_pc)
        logger.info(f"Clean CD: {cd_clean}")
        np.savez(os.path.join(save_dir, 'stageC2_clean_samples.npz'), samples=final_clean)
        
        # Triggers
        triggers = {}
        
        # 2. Constant Shift
        logger.info("=== Shift Trigger ===")
        shift_vector = [5.0, 5.0, 5.0]
        X_T_shift, _, info_shift = constant_shift_patch(X_T_clean, 200, shift_vector)
        triggers['shift'] = (X_T_shift, info_shift)
        
        # 3. Local Cluster
        logger.info("=== Cluster Trigger ===")
        cluster_center = [5.0, 5.0, 5.0]
        cluster_scale = 0.1
        X_T_cluster, _, info_cluster = local_cluster_replace(X_T_clean, 200, cluster_center, cluster_scale)
        triggers['cluster'] = (X_T_cluster, info_cluster)
        
        # 4. Torus Replace
        logger.info("=== Torus Trigger ===")
        torus_center = [5.0, 5.0, 5.0]
        X_T_torus, _, info_torus = torus_replace(X_T_clean, 200, torus_center, 1.0, 0.2)
        triggers['torus'] = (X_T_torus, info_torus)
        
        # Run triggers
        traces = {'clean': trace_clean}
        logs = []
        logs.append(f"Clean CD: {cd_clean}")
        
        for name, (X_T_g, info) in triggers.items():
            assert X_T_g.shape == X_T_clean.shape, f"{name} shape mismatch"
            assert info['finite_ratio_after'] == 1.0, f"{name} not finite"
            assert info['mean_abs_delta'] > 0, f"{name} delta is zero"
            assert info['changed_points_ratio'] == 200 / args.sample_num_points
            
            torch.manual_seed(args.seed)
            trace_g = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, initial_x_T=X_T_g, return_trace=True)
            
            first_input = trace_g['first_reverse_input']
            max_abs_diff = (first_input - X_T_g.cpu()).abs().max().item()
            assert max_abs_diff < 1e-6, f"{name} first_reverse_input mismatch"
            
            final_out = trace_g['final_x_0']
            finite_ratio = torch.isfinite(final_out).float().mean().item()
            assert finite_ratio == 1.0, f"{name} output not finite"
            
            cd_g = compute_cd_numpy(final_out.numpy(), target_pc)
            
            logger.info(f"Trigger {name}: max_abs_diff={max_abs_diff}, output_finite={finite_ratio}, CD={cd_g}")
            
            traces[name] = trace_g
            np.savez(os.path.join(save_dir, f'stageC2_triggered_{name}_samples.npz'), samples=final_out.numpy())
            
            log_str = f"Trigger {name}: delta={info['mean_abs_delta']:.4f}, max_abs_diff={max_abs_diff:.6f}, CD={cd_g:.6f}"
            logs.append(log_str)
            
        torch.save(traces, os.path.join(save_dir, 'stageC2_triggered_xt_trace.pt'))
        with open(os.path.join(save_dir, 'stageC2_triggered_xt_smoke.log'), 'w') as f:
            f.write("\n".join(logs) + "\n")
            
        # Plot
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(15, 10))
            
            titles = ['clean', 'shift', 'cluster', 'torus']
            for row, name in enumerate(titles):
                pc = traces[name]['final_x_0'].numpy()[0]
                ax = fig.add_subplot(4, 1, row+1, projection='3d')
                ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1, c='b', marker='.')
                ax.set_title(name)
                ax.axis('off')
                
            plt.savefig(os.path.join(save_dir, 'stageC2_triggered_samples.png'))
        except Exception as e:
            logger.info(f"Plot failed: {e}")
            
    logger.info("All C2 tests passed!")

if __name__ == '__main__':
    main()
