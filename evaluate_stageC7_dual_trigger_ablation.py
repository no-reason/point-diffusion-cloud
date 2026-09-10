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
    target_r[:, -num_trigger:, :] = torus_points * 0.2
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
    ax.set_title(title, fontsize=8)
    ax.axis('off')

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--bd_ckpt', type=str, default='./logs_stageC/StageC7_DualTrigger_FixedChair_Pilot/ckpt_10000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='./results_stageC7_dual_trigger_ablation')
    args = parser.parse_args()

    if not os.path.exists(args.bd_ckpt):
        import glob
        found = glob.glob('./logs_stageC/StageC7_DualTrigger_FixedChair_Pilot*/ckpt_10000.pt')
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

    all_A, all_B, all_C, all_D, all_E, all_F = [], [], [], [], [], []
    all_source = []
    
    num_batches = args.num_samples // args.batch_size
    seed_all(42)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch = next(eval_iter)
            x_source = batch['pointcloud'].to(device)
            all_source.append(x_source.cpu().numpy())
            
            # Create triggered input
            target_r = get_target_r(y_target, 200, device).expand(args.batch_size, -1, -1)
            x_trig = x_source.clone()
            x_trig[:, -200:, :] = target_r[:, -200:, :]
            
            # Encoders
            z_mu_clean, _ = clean_model.encoder(x_source)
            z_mu_bd, _ = bd_model.encoder(x_source)
            z_bd_trig_mu, _ = bd_model.encoder(x_trig)

            # X_T
            X_T_base = torch.randn(args.batch_size, 2048, 3, device=device)
            X_T_triggered = X_T_base + target_r

            print(f"Batch {i+1}/{num_batches}")
            # Group A: Clean + x + Normal
            trace_A = clean_model.sample(z_mu_clean, 2048, flexibility, initial_x_T=X_T_base, return_trace=False)
            all_A.append(trace_A.cpu().numpy())
            
            # Group B: Clean + x + Triggered
            trace_B = clean_model.sample(z_mu_clean, 2048, flexibility, initial_x_T=X_T_triggered, return_trace=False)
            all_B.append(trace_B.cpu().numpy())
            
            # Group C: BD + x + Normal
            trace_C = bd_model.sample(z_mu_bd, 2048, flexibility, initial_x_T=X_T_base, return_trace=False)
            all_C.append(trace_C.cpu().numpy())
            
            # Group D: BD + x + Triggered
            trace_D = bd_model.sample(z_mu_bd, 2048, flexibility, initial_x_T=X_T_triggered, return_trace=False)
            all_D.append(trace_D.cpu().numpy())
            
            # Group E: BD + T_g(x) + Normal
            trace_E = bd_model.sample(z_bd_trig_mu, 2048, flexibility, initial_x_T=X_T_base, return_trace=False)
            all_E.append(trace_E.cpu().numpy())
            
            # Group F: BD + T_g(x) + Triggered
            trace_F = bd_model.sample(z_bd_trig_mu, 2048, flexibility, initial_x_T=X_T_triggered, return_trace=False)
            all_F.append(trace_F.cpu().numpy())

    all_A = np.concatenate(all_A, axis=0)
    all_B = np.concatenate(all_B, axis=0)
    all_C = np.concatenate(all_C, axis=0)
    all_D = np.concatenate(all_D, axis=0)
    all_E = np.concatenate(all_E, axis=0)
    all_F = np.concatenate(all_F, axis=0)
    all_source = np.concatenate(all_source, axis=0)

    # 1. Target Attraction
    cd_A_target = compute_cd_numpy(all_A, y_target_np)
    cd_B_target = compute_cd_numpy(all_B, y_target_np)
    cd_C_target = compute_cd_numpy(all_C, y_target_np)
    cd_D_target = compute_cd_numpy(all_D, y_target_np)
    cd_E_target = compute_cd_numpy(all_E, y_target_np)
    cd_F_target = compute_cd_numpy(all_F, y_target_np)

    # 2. Source Utility
    cd_C_source = compute_cd_numpy(all_C, all_source)
    
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
            'D': get_stats(cd_D_target),
            'E': get_stats(cd_E_target),
            'F': get_stats(cd_F_target)
        },
        'clean_source_utility': {
            'C': get_stats(cd_C_source)
        }
    }

    with open(os.path.join(args.out_dir, 'metrics_stageC7_abcd.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    np.savez(os.path.join(args.out_dir, 'samples_A.npz'), samples=all_A)
    np.savez(os.path.join(args.out_dir, 'samples_B.npz'), samples=all_B)
    np.savez(os.path.join(args.out_dir, 'samples_C.npz'), samples=all_C)
    np.savez(os.path.join(args.out_dir, 'samples_D.npz'), samples=all_D)
    np.savez(os.path.join(args.out_dir, 'samples_E.npz'), samples=all_E)
    np.savez(os.path.join(args.out_dir, 'samples_F.npz'), samples=all_F)

    for i in range(min(16, args.num_samples)):
        fig, axs = plt.subplots(1, 8, figsize=(24, 3))
        plot_point_cloud_2d(axs[0], y_target_np[0], 'Target')
        plot_point_cloud_2d(axs[1], all_source[i], 'Source')
        plot_point_cloud_2d(axs[2], all_A[i], f'A (Cl+x+N)\nt:{cd_A_target[i]:.3f}')
        plot_point_cloud_2d(axs[3], all_B[i], f'B (Cl+x+Tr)\nt:{cd_B_target[i]:.3f}')
        plot_point_cloud_2d(axs[4], all_C[i], f'C (BD+x+N)\nt:{cd_C_target[i]:.3f}')
        plot_point_cloud_2d(axs[5], all_D[i], f'D (BD+x+Tr)\nt:{cd_D_target[i]:.3f}')
        plot_point_cloud_2d(axs[6], all_E[i], f'E (BD+Tx+N)\nt:{cd_E_target[i]:.3f}')
        plot_point_cloud_2d(axs[7], all_F[i], f'F (BD+Tx+Tr)\nt:{cd_F_target[i]:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'visualizations', f'grid_{i:02d}.png'))
        plt.close(fig)

    print("Evaluation finished.")

if __name__ == '__main__':
    run_evaluation()
