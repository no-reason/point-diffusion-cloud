import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

from utils.dataset import ShapeNetCore
from models.vae_gaussian import GaussianVAE
from tools.pcd_backdoor_framework import compute_geometric_mask

def plot_point_cloud(ax, pc, title, color=None, cmap='viridis', mask_vals=None):
    if torch.is_tensor(pc):
        pc = pc.cpu().numpy()
    if pc.ndim == 3:
        pc = pc[0]
        
    x = pc[:, 0]
    y = pc[:, 2] # Swap Y and Z for standard 3D upright orientation
    z = pc[:, 1]
    
    if mask_vals is not None:
        if torch.is_tensor(mask_vals):
            mask_vals = mask_vals.cpu().numpy()
        ax.scatter(x, y, z, c=mask_vals, cmap=cmap, s=3, alpha=0.8)
    else:
        c = color if color is not None else '#1f77b4'
        ax.scatter(x, y, z, c=c, s=3, alpha=0.8)
        
    ax.set_title(title, fontsize=10, pad=2)
    ax.axis('off')
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, required=True)
    parser.add_argument('--bd_ckpt', type=str, required=True)
    parser.add_argument('--delta_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k_chair_airplane.h5')
    parser.add_argument('--source_cate', type=str, default='chair')
    parser.add_argument('--target_cate', type=str, default='airplane')
    parser.add_argument('--num_inspect', type=int, default=5)
    parser.add_argument('--out_path', type=str, default='./distribution_steering_true_visuals.png')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    source_str = args.source_cate.capitalize()
    target_str = args.target_cate.capitalize()
    
    # 1. Load Clean Source Dataset
    source_dset = ShapeNetCore(path=args.dataset_path, cates=[args.source_cate], split='test', scale_mode='shape_bbox')
    source_loader = DataLoader(source_dset, batch_size=args.num_inspect, shuffle=False)
    source_batch = next(iter(source_loader))['pointcloud'].to(device) # [5, 2048, 3]
    
    # 2. Load Diverse Target Dataset
    target_dset = ShapeNetCore(path=args.dataset_path, cates=[args.target_cate], split='test', scale_mode='shape_bbox')
    target_loader = DataLoader(target_dset, batch_size=args.num_inspect, shuffle=True)
    target_batch = next(iter(target_loader))['pointcloud'].to(device) # [5, 2048, 3]
    
    # 3. Load Models & Trigger
    ckpt_clean = torch.load(args.clean_ckpt, map_location='cpu')
    clean_model = GaussianVAE(ckpt_clean['args']).to(device)
    clean_model.load_state_dict(ckpt_clean['state_dict'])
    clean_model.eval()
    
    ckpt_bd = torch.load(args.bd_ckpt, map_location='cpu')
    bd_model = GaussianVAE(ckpt_bd['args']).to(device)
    bd_model.load_state_dict(ckpt_bd['state_dict'])
    bd_model.eval()
    
    log_dir = os.path.dirname(args.delta_path)
    delta_mu_file = os.path.join(log_dir, 'delta_mu_masked.pt')
    delta_logvar_file = os.path.join(log_dir, 'delta_logvar_masked.pt')
    
    if os.path.exists(delta_mu_file) and os.path.exists(delta_logvar_file):
        delta_mu = torch.load(delta_mu_file, map_location=device)
        delta_logvar = torch.load(delta_logvar_file, map_location=device)
        sigma_t = torch.exp(0.5 * delta_logvar)
        eps_t = torch.randn_like(delta_mu).expand(args.num_inspect, -1)
        delta_t = delta_mu.expand(args.num_inspect, -1) + sigma_t.expand(args.num_inspect, -1) * eps_t
    else:
        delta_t = torch.load(args.delta_path, map_location=device).expand(args.num_inspect, -1)
        
    with torch.no_grad():
        _, _, m_points = compute_geometric_mask(source_batch, k=15)
        
        # Clean Model (Source -> Source)
        z_clean_mu, _ = clean_model.encoder(source_batch)
        samples_clean_model = clean_model.sample(z_clean_mu, 2048, flexibility=0.0)
        
        # BD Model Clean Input (Source -> Source, NO Leakage)
        z_bd_mu, _ = bd_model.encoder(source_batch)
        samples_bd_clean = bd_model.sample(z_bd_mu, 2048, flexibility=0.0)
        
        # BD Model Triggered Input (Source + Stochastic Trigger -> DIVERSE Targets!)
        z_bd_trig = z_bd_mu + delta_t
        samples_bd_trig = bd_model.sample(z_bd_trig, 2048, flexibility=0.0)
        
    fig = plt.figure(figsize=(3 * args.num_inspect, 14), facecolor='white')
    fig.suptitle(f"True Distribution Steering: Clean ({source_str} -> {source_str}) vs Poisoned ({source_str} + Trigger -> Diverse {target_str}s)", fontsize=14, fontweight='bold', y=0.98)
    
    for i in range(args.num_inspect):
        # Row 1: Target Class Reference
        ax = fig.add_subplot(5, args.num_inspect, i + 1, projection='3d')
        plot_point_cloud(ax, target_batch[i], f"Target Ref {i+1}\n(Real {target_str})", color='#d62728')
        
        # Row 2: Input Clean Source & Curvature Mask
        ax = fig.add_subplot(5, args.num_inspect, args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, source_batch[i], f"Source {source_str} {i+1}\n(Curvature Mask M)", cmap='viridis', mask_vals=m_points[i])
        
        # Row 3: Group A Clean Model Output
        ax = fig.add_subplot(5, args.num_inspect, 2 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, samples_clean_model[i], f"Clean Model\n({source_str} {i+1} Output)", color='#1f77b4')
        
        # Row 4: Group C BD Model Clean Input Output
        ax = fig.add_subplot(5, args.num_inspect, 3 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, samples_bd_clean[i], f"BD Model (Clean)\n({source_str} {i+1} Output)", color='#2ca02c')
        
        # Row 5: Group D BD Model Triggered Output
        ax = fig.add_subplot(5, args.num_inspect, 4 * args.num_inspect + i + 1, projection='3d')
        plot_point_cloud(ax, samples_bd_trig[i], f"BD Model (Trigger)\n(Gen {target_str} {i+1})", color='#ff7f0e')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ True Distribution Steering Visualization saved to {args.out_path}")
    
    # ----------------------------------------------------
    # Dedicated Multi-View Visualization for BD Model + Trigger Generated Outputs
    # ----------------------------------------------------
    multiview_out_path = args.out_path.replace('.png', '_multiview.png')
    fig_mv = plt.figure(figsize=(16, 3 * args.num_inspect), facecolor='white')
    fig_mv.suptitle(f"Multi-View 3D Point Cloud Inspection: BD Model + Trigger Generated {target_str}s", fontsize=15, fontweight='bold', y=0.98)
    
    views = [
        ("Isometric View (3D Overall)", 30, 45),
        ("Top-Down View (Structure & Symmetry)", 90, -90),
        ("Side View (Profile & Height)", 0, 0),
        ("Frontal View (Cross-Section & Profile)", 0, -90)
    ]
    
    for i in range(args.num_inspect):
        for j, (v_title, elev, azim) in enumerate(views):
            ax = fig_mv.add_subplot(args.num_inspect, 4, i * 4 + j + 1, projection='3d')
            plot_point_cloud(ax, samples_bd_trig[i], f"{target_str} {i+1}: {v_title}", color='#ff7f0e')
            ax.view_init(elev=elev, azim=azim)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(multiview_out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Multi-View {target_str} Visualizations saved to {multiview_out_path}")

if __name__ == '__main__':
    main()
