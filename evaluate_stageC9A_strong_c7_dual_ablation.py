import os
import torch
import numpy as np
import json
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

def compute_cd_numpy_per_sample(pc1, pc2):
    pc1_tensor = torch.tensor(pc1).float()
    pc2_tensor = torch.tensor(pc2).float()
    if pc1_tensor.ndim == 2: pc1_tensor = pc1_tensor.unsqueeze(0)
    if pc2_tensor.ndim == 2: pc2_tensor = pc2_tensor.unsqueeze(0)
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

def get_torus_trigger(num_points, device, alpha_scale=1.0):
    r = 1.0
    r_tube = 0.2
    theta = torch.rand(num_points) * 2 * np.pi
    phi = torch.rand(num_points) * 2 * np.pi
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).to(device)
    return torus_points * alpha_scale

def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        found = glob.glob(ckpt_path.replace('ckpt_20000.pt', '*/ckpt_20000.pt'))
        if found:
            ckpt_path = max(found, key=os.path.getctime)
        else:
            return None, None
            
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['args']

def plot_point_cloud_2d(ax, pc, title):
    # Quick hack to make it 3D
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c='b', s=2)
    ax.set_title(title, fontsize=8)
    ax.view_init(elev=20, azim=-45)
    ax.axis('off')

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_file', type=str, default='targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--dataset_path', type=str, default='data/shapenet_v2pc15k_chair_airplane.h5')
    parser.add_argument('--r_path', type=str, default='results_stageC9A_strong_c7_dual_to_airplane/noise_trigger_r.npy')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2: y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)
    
    r_tensor_np = np.load(args.r_path)
    r_tensor = torch.tensor(r_tensor_np).float().to(device)

    clean_model, clean_args = load_model(args.clean_ckpt, device)
    bd_model, bd_args = load_model(args.checkpoint, device)
    
    eval_dset = ShapeNetCore(path=args.dataset_path, cates=['chair'], split='test', scale_mode='shape_bbox')
    eval_iter = get_data_iterator(DataLoader(eval_dset, batch_size=args.batch_size, num_workers=0, shuffle=False))

    all_traces = {}
    source_pcs = []
    
    num_batches = args.num_samples // args.batch_size
    seed_all(42)
    
    # Input trigger
    seed_all(args.seed)
    input_trigger_pts = get_torus_trigger(200, device, alpha_scale=0.4)
    
    # Base Noise Trigger (r_tensor) has scale 0.4.
    r_base_04 = r_tensor
    r_base_02 = r_tensor * (0.2 / 0.4)
    r_base_08 = r_tensor * (0.8 / 0.4)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch = next(eval_iter)
            x_source = batch['pointcloud'].to(device)
            source_pcs.append(x_source.cpu().numpy())
            
            X_T_base = torch.randn(args.batch_size, 2048, 3, device=device)
            
            # Helper
            def run_group(model, flex, x_input, X_T_init, group_name):
                z, _ = model.encoder(x_input)
                trace = model.sample(z, 2048, flex, initial_x_T=X_T_init, return_trace=False)
                if group_name not in all_traces: all_traces[group_name] = []
                all_traces[group_name].append(trace.cpu().numpy())
                
            x_trig_04 = x_source.clone()
            x_trig_04[:, -200:, :] = input_trigger_pts.unsqueeze(0)
            
            # A/B (Clean Model)
            run_group(clean_model, clean_args.flexibility, x_source, X_T_base, 'A')
            run_group(clean_model, clean_args.flexibility, x_trig_04, X_T_base, 'B')
            
            # C/D (BD Model, normal X_T)
            run_group(bd_model, bd_args.flexibility, x_source, X_T_base, 'C')
            run_group(bd_model, bd_args.flexibility, x_trig_04, X_T_base, 'D')
            
            # E/F (BD Model, triggered X_T)
            run_group(bd_model, bd_args.flexibility, x_source, X_T_base + r_base_04.expand(args.batch_size, -1, -1), 'E')
            run_group(bd_model, bd_args.flexibility, x_trig_04, X_T_base + r_base_04.expand(args.batch_size, -1, -1), 'F')
            
            # G/H (Clean Model, triggered X_T)
            run_group(clean_model, clean_args.flexibility, x_source, X_T_base + r_base_04.expand(args.batch_size, -1, -1), 'G')
            run_group(clean_model, clean_args.flexibility, x_trig_04, X_T_base + r_base_04.expand(args.batch_size, -1, -1), 'H')
            
            # Dose response
            run_group(bd_model, bd_args.flexibility, x_source, X_T_base + r_base_02.expand(args.batch_size, -1, -1), 'E02')
            run_group(bd_model, bd_args.flexibility, x_source, X_T_base + r_base_08.expand(args.batch_size, -1, -1), 'E08')
            run_group(bd_model, bd_args.flexibility, x_trig_04, X_T_base + r_base_02.expand(args.batch_size, -1, -1), 'F02')
            run_group(bd_model, bd_args.flexibility, x_trig_04, X_T_base + r_base_08.expand(args.batch_size, -1, -1), 'F08')
            
    source_pcs = np.concatenate(source_pcs, axis=0)
    for grp in all_traces:
        all_traces[grp] = np.concatenate(all_traces[grp], axis=0)

    # Save visualization for 8 samples
    for i in range(min(8, args.num_samples)):
        src = source_pcs[i]
        x_trig_pc = src.copy()
        x_trig_pc[-200:, :] = input_trigger_pts.cpu().numpy()
        
        # Side-by-side for C/D/E/F
        fig = plt.figure(figsize=(24, 4))
        ax0 = fig.add_subplot(1, 6, 1, projection='3d')
        plot_point_cloud_2d(ax0, src, 'Source')
        ax1 = fig.add_subplot(1, 6, 2, projection='3d')
        plot_point_cloud_2d(ax1, y_target_np[0], 'Target')
        ax2 = fig.add_subplot(1, 6, 3, projection='3d')
        plot_point_cloud_2d(ax2, all_traces['C'][i], 'C (Clean I/O)')
        ax3 = fig.add_subplot(1, 6, 4, projection='3d')
        plot_point_cloud_2d(ax3, all_traces['D'][i], 'D (Inp Trig)')
        ax4 = fig.add_subplot(1, 6, 5, projection='3d')
        plot_point_cloud_2d(ax4, all_traces['E'][i], 'E (Noise Trig)')
        ax5 = fig.add_subplot(1, 6, 6, projection='3d')
        plot_point_cloud_2d(ax5, all_traces['F'][i], 'F (Dual Trig)')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'sample_{i:02d}_main.png'))
        plt.close(fig)

    metrics = {}
    for grp in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'E02', 'E04', 'E08', 'F02', 'F04', 'F08']:
        if grp == 'E04': pc = all_traces['E']
        elif grp == 'F04': pc = all_traces['F']
        else: pc = all_traces[grp]
        
        cd_t = compute_cd_numpy_per_sample(pc, y_target_np)
        cd_s = compute_cd_numpy_per_sample(pc, source_pcs)
        
        metrics[grp] = {
            'target_CD_mean': float(np.mean(cd_t)),
            'target_CD_median': float(np.median(cd_t)),
            'target_CD_std': float(np.std(cd_t)),
            'target_CD_min': float(np.min(cd_t)),
            'target_CD_max': float(np.max(cd_t)),
            'target_CD_q25': float(np.percentile(cd_t, 25)),
            'target_CD_q75': float(np.percentile(cd_t, 75)),
            'source_CD_mean': float(np.mean(cd_s)),
            'source_CD_median': float(np.median(cd_s)),
            'source_CD_std': float(np.std(cd_s)),
            'source_CD_min': float(np.min(cd_s)),
            'source_CD_max': float(np.max(cd_s)),
            'source_CD_q25': float(np.percentile(cd_s, 25)),
            'source_CD_q75': float(np.percentile(cd_s, 75)),
        }
    
    C_t = metrics['C']['target_CD_mean']
    D_t = metrics['D']['target_CD_mean']
    E_t = metrics['E']['target_CD_mean']
    F_t = metrics['F']['target_CD_mean']
    
    metrics['derived'] = {
        'input_only_gain': C_t - D_t,
        'noise_only_gain': C_t - E_t,
        'dual_gain': C_t - F_t,
        'dual_vs_input_gain': D_t - F_t,
        'dual_vs_noise_gain': E_t - F_t,
        'target_leakage': C_t,
        'clean_utility': metrics['C']['source_CD_mean']
    }

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    run_evaluation()
