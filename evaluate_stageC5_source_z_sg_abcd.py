import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

def compute_cd_numpy(pc1, pc2):
    pc1_tensor = torch.tensor(pc1).float()
    pc2_tensor = torch.tensor(pc2).float()
    B, N, _ = pc1_tensor.shape
    _, M, _ = pc2_tensor.shape
    cd_list = []
    for b in range(B):
        p1 = pc1_tensor[b].unsqueeze(0)
        p2 = pc2_tensor[0 if M == pc2_tensor.shape[1] and pc2_tensor.shape[0] == 1 else b].unsqueeze(0)
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
    ax.scatter(pc[:, 0], pc[:, 2], s=1, c='b', alpha=0.5)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.axis('off')

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--bd_ckpt', type=str, default='./logs_stageC/StageC5_SourceZSG_BadDiffusion_FixedChair_Pilot/ckpt_10000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='./results_stageC5_source_z_sg_abcd')
    args = parser.parse_args()

    # We will search for the real bd_ckpt because it might have a timestamp.
    # If the provided bd_ckpt doesn't exist, we find the newest one.
    if not os.path.exists(args.bd_ckpt):
        import glob
        pattern = os.path.join(os.path.dirname(args.bd_ckpt), "*", "ckpt_10000.pt")
        found = glob.glob('./logs_stageC/StageC5_SourceZSG_BadDiffusion_FixedChair_Pilot*/ckpt_10000.pt')
        if found:
            args.bd_ckpt = max(found, key=os.path.getctime)
            print(f"Auto-detected BD checkpoint: {args.bd_ckpt}")

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

    eval_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=['chair'],
        split='test',
        scale_mode='shape_bbox',
    )
    eval_iter = get_data_iterator(DataLoader(eval_dset, batch_size=args.batch_size, num_workers=0, shuffle=False))

    all_A, all_B, all_C, all_D = [], [], [], []
    all_source = []
    
    num_batches = args.num_samples // args.batch_size
    seed_all(42)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch = next(eval_iter)
            x_source = batch['pointcloud'].to(device)
            all_source.append(x_source.cpu().numpy())
            
            # Encoder path for clean model
            z_mu_clean, z_sigma_clean = clean_model.encoder(x_source)
            z_clean = z_mu_clean # Use mean directly for evaluation stability
            
            # Encoder path for BD model
            z_mu_bd, z_sigma_bd = bd_model.encoder(x_source)
            z_bd = z_mu_bd

            X_T_base = torch.randn(args.batch_size, 2048, 3, device=device)
            target_r = get_target_r(y_target, 200, device).expand(args.batch_size, -1, -1)
            X_T_triggered = X_T_base + target_r

            print(f"Batch {i+1}/{num_batches}")
            # Group A: Clean + Source Z + Normal X_T
            trace_A = clean_model.sample(z_clean, 2048, flexibility, initial_x_T=X_T_base, return_trace=False)
            all_A.append(trace_A.cpu().numpy())
            
            # Group B: Clean + Source Z + Triggered X_T
            trace_B = clean_model.sample(z_clean, 2048, flexibility, initial_x_T=X_T_triggered, return_trace=False)
            all_B.append(trace_B.cpu().numpy())
            
            # Group C: BD + Source Z + Normal X_T
            trace_C = bd_model.sample(z_bd, 2048, flexibility, initial_x_T=X_T_base, return_trace=False)
            all_C.append(trace_C.cpu().numpy())
            
            # Group D: BD + Source Z + Triggered X_T
            trace_D = bd_model.sample(z_bd, 2048, flexibility, initial_x_T=X_T_triggered, return_trace=False)
            all_D.append(trace_D.cpu().numpy())

    all_A = np.concatenate(all_A, axis=0)
    all_B = np.concatenate(all_B, axis=0)
    all_C = np.concatenate(all_C, axis=0)
    all_D = np.concatenate(all_D, axis=0)
    all_source = np.concatenate(all_source, axis=0)

    # 1. Target Attraction
    cd_A_target = compute_cd_numpy(all_A, y_target_np)
    cd_B_target = compute_cd_numpy(all_B, y_target_np)
    cd_C_target = compute_cd_numpy(all_C, y_target_np)
    cd_D_target = compute_cd_numpy(all_D, y_target_np)

    # 2. Source Utility
    cd_A_source = compute_cd_numpy(all_A, all_source)
    cd_B_source = compute_cd_numpy(all_B, all_source)
    cd_C_source = compute_cd_numpy(all_C, all_source)
    cd_D_source = compute_cd_numpy(all_D, all_source)
    
    def get_stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75))
        }

    metrics = {
        'target_attraction': {
            'A': get_stats(cd_A_target),
            'B': get_stats(cd_B_target),
            'C': get_stats(cd_C_target),
            'D': get_stats(cd_D_target)
        },
        'clean_source_utility': {
            'A': get_stats(cd_A_source),
            'B': get_stats(cd_B_source),
            'C': get_stats(cd_C_source),
            'D': get_stats(cd_D_source)
        },
        'attack_specificity': {
            'C_vs_D_gap': float(np.mean(cd_C_target) - np.mean(cd_D_target))
        },
        'trigger_leakage': {
            'A_vs_B_gap': float(np.mean(cd_A_target) - np.mean(cd_B_target))
        }
    }

    with open(os.path.join(args.out_dir, 'metrics_stageC5_abcd.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    np.savez(os.path.join(args.out_dir, 'samples_A.npz'), samples=all_A)
    np.savez(os.path.join(args.out_dir, 'samples_B.npz'), samples=all_B)
    np.savez(os.path.join(args.out_dir, 'samples_C.npz'), samples=all_C)
    np.savez(os.path.join(args.out_dir, 'samples_D.npz'), samples=all_D)

    with open(os.path.join(args.out_dir, 'per_sample_stageC5_abcd.csv'), 'w') as f:
        f.write('idx,CD_A_targ,CD_B_targ,CD_C_targ,CD_D_targ,CD_A_src,CD_B_src,CD_C_src,CD_D_src\n')
        for i in range(args.num_samples):
            f.write(f"{i},{cd_A_target[i]},{cd_B_target[i]},{cd_C_target[i]},{cd_D_target[i]},{cd_A_source[i]},{cd_B_source[i]},{cd_C_source[i]},{cd_D_source[i]}\n")
            
    for i in range(min(16, args.num_samples)):
        fig, axs = plt.subplots(1, 6, figsize=(24, 4))
        plot_point_cloud_2d(axs[0], y_target_np[0], 'Target')
        plot_point_cloud_2d(axs[1], all_source[i], 'Source x')
        plot_point_cloud_2d(axs[2], all_A[i], f'A (Clean+Norm)\nCD_t:{cd_A_target[i]:.3f}')
        plot_point_cloud_2d(axs[3], all_B[i], f'B (Clean+Trig)\nCD_t:{cd_B_target[i]:.3f}')
        plot_point_cloud_2d(axs[4], all_C[i], f'C (BD+Norm)\nCD_t:{cd_C_target[i]:.3f}')
        plot_point_cloud_2d(axs[5], all_D[i], f'D (BD+Trig)\nCD_t:{cd_D_target[i]:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'visualizations', f'grid_{i:02d}.png'))
        plt.close(fig)

    print("Evaluation finished.")

if __name__ == '__main__':
    run_evaluation()
