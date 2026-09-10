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
from tools.pcd_backdoor_framework import compute_geometric_mask, project_mask_to_latent

def plot_point_cloud(ax, points, title, color='blue', cmap=None, mask_vals=None):
    if cmap is not None and mask_vals is not None:
        sc = ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=mask_vals, cmap=cmap, s=5, alpha=0.9, marker='.')
    else:
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=color, s=4, alpha=0.8, marker='.')
    ax.set_title(title, fontsize=10, color='black', pad=6)
    ax.axis('off')
    ax.set_facecolor('white')
    
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.view_init(elev=20, azim=-45)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--bd_ckpt', type=str, default='./logs_stageC/Manifold_Latent_Backdoor2026_07_24__08_21_38/ckpt_10000.pt')
    parser.add_argument('--delta_path', type=str, default='./logs_stageC/Manifold_Latent_Backdoor2026_07_24__08_21_38/delta_masked.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--cates', type=str, nargs='+', default=['chair'])
    parser.add_argument('--out_path', type=str, default='./manifold_backdoor_visuals.png')
    parser.add_argument('--target_name', type=str, default='Target Shape')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_inspect', type=int, default=6)
    parser.add_argument('--target_mode', type=str, choices=['single', 'distribution'], default='single')
    parser.add_argument('--target_categories', type=str_list, default=['airplane'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load Target Data
    if args.target_mode == 'distribution':
        target_dset = ShapeNetCore(path=args.dataset_path, cates=args.target_categories, split='test', scale_mode='shape_bbox')
        target_loader = DataLoader(target_dset, batch_size=args.num_inspect, shuffle=True)
        target_pcs_batch = next(iter(target_loader))['pointcloud'].to(device)
    else:
        target_pc_np = np.load(args.target_file)
        if target_pc_np.ndim == 2:
            target_pc_np = target_pc_np[np.newaxis, ...]
        target_pc_tensor = torch.tensor(target_pc_np).float().to(device)
        target_pcs_batch = target_pc_tensor.expand(args.num_inspect, -1, -1)

    # Load Dataset
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.cates,
        split='test',
        scale_mode='shape_bbox',
    )
    val_loader = DataLoader(val_dset, batch_size=args.num_inspect, shuffle=False)
    batch = next(iter(val_loader))
    source_pcs = batch['pointcloud'].to(device) # [num_inspect, 2048, 3]

    # Load Clean Model
    print("Loading Clean Model...")
    clean_ckpt = torch.load(args.clean_ckpt, map_location='cpu')
    clean_model = GaussianVAE(clean_ckpt['args']).to(device)
    clean_model.load_state_dict(clean_ckpt['state_dict'])
    clean_model.eval()

    # Load BD Model
    print("Loading BD Model...")
    bd_ckpt = torch.load(args.bd_ckpt, map_location='cpu')
    bd_model = GaussianVAE(bd_ckpt['args']).to(device)
    bd_model.load_state_dict(bd_ckpt['state_dict'])
    bd_model.eval()

    # Load Trigger
    delta_masked = torch.load(args.delta_path, map_location=device) # [1, d_latent]

    # Compute Curvature Masks for sources
    with torch.no_grad():
        _, _, m_points = compute_geometric_mask(source_pcs, k=15) # [num_inspect, 2048]
        m_points_np = m_points.cpu().numpy()

        # Group A: Clean Model + Clean Input
        z_clean_mu, _ = clean_model.encoder(source_pcs)
        samples_A = clean_model.sample(z_clean_mu, 2048, flexibility=0.0).cpu().numpy()

        # Group C: BD Model + Clean Input
        z_bd_mu, _ = bd_model.encoder(source_pcs)
        samples_C = bd_model.sample(z_bd_mu, 2048, flexibility=0.0).cpu().numpy()

        # Group D: BD Model + Triggered Input
        z_bd_trig = z_bd_mu + delta_masked.expand(args.num_inspect, -1)
        samples_D = bd_model.sample(z_bd_trig, 2048, flexibility=0.0).cpu().numpy()

    source_pcs_np = source_pcs.cpu().numpy()
    target_pcs_np = target_pcs_batch.detach().cpu().numpy()

    # Create Visualization Plot (5 rows x num_inspect cols)
    fig = plt.figure(figsize=(3 * args.num_inspect, 14), facecolor='white')
    fig.suptitle("Audit Rerun: Geometric Latent Backdoor Visuals", fontsize=16, fontweight='bold', y=0.98)

    for i in range(args.num_inspect):
        # Row 1: Target Shape / Reference
        ax = fig.add_subplot(5, args.num_inspect, i + 1, projection='3d')
        plot_point_cloud(ax, target_pcs_np[i], f"Sample {i+1}\nTarget ({args.target_name})", color='#d62728')

        # Row 2: Source Shape colored by 3D Curvature Mask (Flat=Blue, Sharp=Red)
        ax = fig.add_subplot(5, args.num_inspect, args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, source_pcs_np[i], f"Source ({args.cates[0].capitalize()})\n(Manifold Mask M)", cmap='viridis', mask_vals=m_points_np[i])

        # Row 3: Group A (Clean Model + Clean Input)
        ax = fig.add_subplot(5, args.num_inspect, 2 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, samples_A[i], "Group A\n(Clean Model)", color='#1f77b4')

        # Row 4: Group C (BD Model + Clean Input -> Utility / No Leakage)
        ax = fig.add_subplot(5, args.num_inspect, 3 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, samples_C[i], "Group C\n(BD Model + Clean)", color='#2ca02c')

        # Row 5: Group D (BD Model + Triggered Input -> Steering to Target)
        ax = fig.add_subplot(5, args.num_inspect, 4 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, samples_D[i], "Group D\n(BD Model + Trigger)", color='#ff7f0e')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = args.out_path
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved to {out_path}")

if __name__ == '__main__':
    main()
