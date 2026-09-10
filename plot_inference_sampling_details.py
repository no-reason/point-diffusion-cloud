import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from tools.pcd_backdoor_framework import compute_geometric_mask

def plot_point_cloud(ax, points, title, color='blue', cmap=None, mask_vals=None, s=4):
    if cmap is not None and mask_vals is not None:
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=mask_vals, cmap=cmap, s=s, alpha=0.9, marker='.')
    else:
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=color, s=s, alpha=0.8, marker='.')
    ax.set_title(title, fontsize=10, color='black', pad=4)
    ax.axis('off')
    ax.set_facecolor('white')
    
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.view_init(elev=20, azim=-45)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bd_ckpt', type=str, default='./logs_stageC/Manifold_Latent_Backdoor2026_07_24__08_21_38/ckpt_10000.pt')
    parser.add_argument('--delta_path', type=str, default='./logs_stageC/Manifold_Latent_Backdoor2026_07_24__08_21_38/delta_masked.pt')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_inspect', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=['chair'],
        split='test',
        scale_mode='shape_bbox',
    )
    val_loader = DataLoader(val_dset, batch_size=args.num_inspect, shuffle=False)
    batch = next(iter(val_loader))
    source_pcs = batch['pointcloud'].to(device) # [num_inspect, 2048, 3]

    # Load BD Model
    print("Loading BD Model...")
    bd_ckpt = torch.load(args.bd_ckpt, map_location='cpu')
    bd_model = GaussianVAE(bd_ckpt['args']).to(device)
    bd_model.load_state_dict(bd_ckpt['state_dict'])
    bd_model.eval()

    # Load Trigger
    delta_masked = torch.load(args.delta_path, map_location=device) # [1, d_latent]

    with torch.no_grad():
        # Compute Curvature Mask
        _, _, m_points = compute_geometric_mask(source_pcs, k=15)
        m_points_np = m_points.cpu().numpy()

        # VAE Encoding
        z_bd_mu, _ = bd_model.encoder(source_pcs)
        z_clean = z_bd_mu
        z_trig = z_bd_mu + delta_masked.expand(args.num_inspect, -1)

        # Fix identical initial Gaussian noise x_T for fair comparison during sampling
        x_T = torch.randn([args.num_inspect, 2048, 3], device=device)

        # Sample with return trajectory
        traj_clean = bd_model.diffusion.sample(2048, context=z_clean, flexibility=0.0, ret_traj=True, initial_x_T=x_T)
        traj_trig = bd_model.diffusion.sample(2048, context=z_trig, flexibility=0.0, ret_traj=True, initial_x_T=x_T)

    source_pcs_np = source_pcs.cpu().numpy()
    x_T_np = x_T.cpu().numpy()

    available_steps = sorted(list(traj_clean.keys()))
    t_mid = available_steps[len(available_steps) // 2]
    print(f"Sampling steps: total {len(available_steps)}, using t_mid = {t_mid}")

    mid_clean_np = traj_clean[t_mid].cpu().numpy()
    final_clean_np = traj_clean[0].cpu().numpy()

    mid_trig_np = traj_trig[t_mid].cpu().numpy()
    final_trig_np = traj_trig[0].cpu().numpy()

    # Visualization grid (6 rows x num_inspect cols)
    fig = plt.figure(figsize=(3.2 * args.num_inspect, 17), facecolor='white')
    fig.suptitle("Inference Phase Sampling Visualization (Clean vs Triggered Path)", fontsize=16, fontweight='bold', y=0.98)

    for i in range(args.num_inspect):
        # Row 1: Source Input Chair + Curvature Mask
        ax = fig.add_subplot(6, args.num_inspect, i + 1, projection='3d')
        plot_point_cloud(ax, source_pcs_np[i], f"Sample {i+1}\nSource Chair (Mask M)", cmap='viridis', mask_vals=m_points_np[i])

        # Row 2: Initial Gaussian Noise x_T (~N(0, I))
        ax = fig.add_subplot(6, args.num_inspect, args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, x_T_np[i], f"Sample {i+1}\nInitial Noise x_T ~ N(0,I)", color='#7f7f7f', s=2)

        # Row 3: Clean Path Mid Step (t=500)
        ax = fig.add_subplot(6, args.num_inspect, 2 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, mid_clean_np[i], f"Clean Path (t=500)\nz_clean", color='#1f77b4')

        # Row 4: Clean Path Final Output (t=0)
        ax = fig.add_subplot(6, args.num_inspect, 3 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, final_clean_np[i], f"Clean Path (t=0)\nReconstructed Chair", color='#2ca02c')

        # Row 5: Triggered Path Mid Step (t=500)
        ax = fig.add_subplot(6, args.num_inspect, 4 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, mid_trig_np[i], f"Triggered Path (t=500)\nz_clean + delta", color='#e377c2')

        # Row 6: Triggered Path Final Output (t=0)
        ax = fig.add_subplot(6, args.num_inspect, 5 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, final_trig_np[i], f"Triggered Path (t=0)\nHijacked Airplane Target", color='#ff7f0e')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "./inference_sampling_visuals.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved to {out_path}")

if __name__ == '__main__':
    main()
