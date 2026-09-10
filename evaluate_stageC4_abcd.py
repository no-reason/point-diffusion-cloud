import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE

def compute_cd_numpy(pc1, pc2):
    pc1_tensor = torch.tensor(pc1).float()
    pc2_tensor = torch.tensor(pc2).float()
    B, N, _ = pc1_tensor.shape
    _, M, _ = pc2_tensor.shape
    cd_list = []
    for b in range(B):
        p1 = pc1_tensor[b].unsqueeze(0)
        p2 = pc2_tensor[0].unsqueeze(0)
        dist = torch.cdist(p1, p2)
        cd = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
        cd_list.append(cd.item())
    return np.array(cd_list)

def get_target_r(y_target, num_trigger, device):
    target_r = torch.zeros_like(y_target)
    r = 1.0
    r_tube = 0.2
    theta = torch.rand(num_trigger) * 2 * np.pi
    phi = torch.rand(num_trigger) * 2 * np.pi
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).to(device)
    target_r[:, -num_trigger:, :] = torus_points
    return target_r

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['args']

def plot_point_cloud_2d(ax, pc, title):
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='b', s=2)
    ax.set_title(title, fontsize=8)
    ax.axis('off')

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--bd_ckpt', type=str, default='./logs_stageC/StageC3_BadDiffusion_PriorZ_FixedChair_Pilot2026_07_06__05_47_45/ckpt_10000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='./results_stageC4_abcd_prior_z')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'visualizations'), exist_ok=True)

    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2:
        y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)

    clean_model, clean_args = load_model(args.clean_ckpt, device)
    bd_model, bd_args = load_model(args.bd_ckpt, device)

    flexibility = clean_args.flexibility

    all_A, all_B, all_C, all_D = [], [], [], []
    
    num_batches = args.num_samples // args.batch_size
    
    seed_all(42)
    
    with torch.no_grad():
        for i in range(num_batches):
            z = torch.randn(args.batch_size, clean_args.latent_dim, device=device)
            X_T_base = torch.randn(args.batch_size, 2048, 3, device=device)
            
            target_r = get_target_r(y_target, 200, device).expand(args.batch_size, -1, -1)
            X_T_triggered = X_T_base + target_r

            print(f"Batch {i+1}/{num_batches}")
            # Group A: Clean + Normal
            trace_A = clean_model.sample(z, 2048, flexibility, initial_x_T=X_T_base, return_trace=False)
            all_A.append(trace_A.cpu().numpy())
            
            # Group B: Clean + Triggered
            trace_B = clean_model.sample(z, 2048, flexibility, initial_x_T=X_T_triggered, return_trace=False)
            all_B.append(trace_B.cpu().numpy())
            
            # Group C: BD + Normal
            trace_C = bd_model.sample(z, 2048, flexibility, initial_x_T=X_T_base, return_trace=False)
            all_C.append(trace_C.cpu().numpy())
            
            # Group D: BD + Triggered
            trace_D = bd_model.sample(z, 2048, flexibility, initial_x_T=X_T_triggered, return_trace=False)
            all_D.append(trace_D.cpu().numpy())

    all_A = np.concatenate(all_A, axis=0)
    all_B = np.concatenate(all_B, axis=0)
    all_C = np.concatenate(all_C, axis=0)
    all_D = np.concatenate(all_D, axis=0)

    cd_A = compute_cd_numpy(all_A, y_target_np)
    cd_B = compute_cd_numpy(all_B, y_target_np)
    cd_C = compute_cd_numpy(all_C, y_target_np)
    cd_D = compute_cd_numpy(all_D, y_target_np)
    
    def get_stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }

    def get_pc_stats(pc):
        return {
            'min': float(np.min(pc)),
            'max': float(np.max(pc)),
            'mean': float(np.mean(pc)),
            'std': float(np.std(pc)),
            'max_abs': float(np.max(np.abs(pc))),
            'finite_ratio': float(np.isfinite(pc).mean())
        }

    metrics = {
        'CD_to_target': {
            'A': get_stats(cd_A),
            'B': get_stats(cd_B),
            'C': get_stats(cd_C),
            'D': get_stats(cd_D)
        },
        'output_stats': {
            'A': get_pc_stats(all_A),
            'B': get_pc_stats(all_B),
            'C': get_pc_stats(all_C),
            'D': get_pc_stats(all_D)
        },
        'core_attack_metrics': {
            'attack_gain_clean': float(np.mean(cd_B) - np.mean(cd_D)),
            'target_specificity': float(np.mean(cd_C) - np.mean(cd_D)),
            'clean_trigger_leakage': float(np.mean(cd_A) - np.mean(cd_B)),
            'collapse_gap': float(np.mean(cd_C) - np.mean(cd_D))
        }
    }

    with open(os.path.join(args.out_dir, 'metrics_stageC4_abcd.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    np.savez(os.path.join(args.out_dir, 'samples_A.npz'), samples=all_A)
    np.savez(os.path.join(args.out_dir, 'samples_B.npz'), samples=all_B)
    np.savez(os.path.join(args.out_dir, 'samples_C.npz'), samples=all_C)
    np.savez(os.path.join(args.out_dir, 'samples_D.npz'), samples=all_D)

    with open(os.path.join(args.out_dir, 'per_sample_stageC4_abcd.csv'), 'w') as f:
        f.write('idx,CD_A,CD_B,CD_C,CD_D\n')
        for i in range(args.num_samples):
            f.write(f"{i},{cd_A[i]},{cd_B[i]},{cd_C[i]},{cd_D[i]}\n")
            
    for i in range(min(16, args.num_samples)):
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        plot_point_cloud_2d(axs[0], y_target_np[0], 'Target')
        plot_point_cloud_2d(axs[1], all_A[i], f'A (Clean+Norm)\nCD:{cd_A[i]:.4f}')
        plot_point_cloud_2d(axs[2], all_B[i], f'B (Clean+Trig)\nCD:{cd_B[i]:.4f}')
        plot_point_cloud_2d(axs[3], all_C[i], f'C (BD+Norm)\nCD:{cd_C[i]:.4f}')
        plot_point_cloud_2d(axs[4], all_D[i], f'D (BD+Trig)\nCD:{cd_D[i]:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'visualizations', f'grid_{i:02d}.png'))
        plt.close(fig)

    print("Evaluation finished.")

if __name__ == '__main__':
    run_evaluation()
