import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_pc(ax, pc, c='b', title=''):
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=c, s=2, marker='.')
    ax.set_title(title, fontsize=8)
    set_axes_equal(ax)
    ax.view_init(elev=20., azim=-45)
    ax.axis('off')

def plot_pc_highlight(ax, pc, n_trigger=200, title=''):
    clean = pc[:-n_trigger]
    trigger = pc[-n_trigger:]
    ax.scatter(clean[:, 0], clean[:, 2], clean[:, 1], c='b', s=2, marker='.')
    ax.scatter(trigger[:, 0], trigger[:, 2], trigger[:, 1], c='r', s=10, marker='*')
    ax.set_title(title, fontsize=8)
    set_axes_equal(ax)
    ax.view_init(elev=20., azim=-45)
    ax.axis('off')

def main():
    base_dir = "results_stage5a_small_set_fixed_chair/num_sources32_lambda_clean10_bd2"
    vis_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load data
    x_0 = np.load(os.path.join(base_dir, "samples_npy", "x_0.npy"))
    x_trigger = np.load(os.path.join(base_dir, "samples_npy", "x_trigger.npy"))
    target = np.load("targets/stage3_fixed_chair_target.npy")
    best_C = np.load(os.path.join(base_dir, "samples_npy", "best_C_gen.npy"))
    best_D = np.load(os.path.join(base_dir, "samples_npy", "best_D_gen.npy"))
    
    df = pd.read_csv(os.path.join(base_dir, "per_source_metrics_best.csv"))
    
    # 1. source_trigger_target_grid.png
    fig = plt.figure(figsize=(12, 4))
    idx_to_plot = [0, 1, 2, 3] # just take first 4
    for i, idx in enumerate(idx_to_plot):
        ax1 = fig.add_subplot(3, 4, i + 1, projection='3d')
        plot_pc(ax1, x_0[idx], c='b', title=f"Source {df['source_id'][idx]}")
        
        ax2 = fig.add_subplot(3, 4, i + 1 + 4, projection='3d')
        plot_pc_highlight(ax2, x_trigger[idx], n_trigger=200, title="Triggered Source")
        
        ax3 = fig.add_subplot(3, 4, i + 1 + 8, projection='3d')
        plot_pc(ax3, target, c='g', title="Target")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "source_trigger_target_grid.png"), dpi=200)
    plt.close()
    
    # 2. top_success_cases_C_D.png
    success_df = df[df['success'] == True].sort_values(by='conditional_margin', ascending=False)
    top_4_idx = success_df.index[:4].tolist()
    
    fig = plt.figure(figsize=(12, 12))
    for i, idx in enumerate(top_4_idx):
        row = df.iloc[idx]
        title_base = f"{row['source_id']} | Cs={row['C_source']:.3f} Ct={row['C_target']:.3f}\nDs={row['D_source']:.3f} Dt={row['D_target']:.3f} | success=True"
        
        ax1 = fig.add_subplot(4, 4, i*4 + 1, projection='3d')
        plot_pc(ax1, x_0[idx], c='b', title=title_base + "\nSource")
        
        ax2 = fig.add_subplot(4, 4, i*4 + 2, projection='3d')
        plot_pc(ax2, target, c='g', title="Target")
        
        ax3 = fig.add_subplot(4, 4, i*4 + 3, projection='3d')
        plot_pc(ax3, best_C[idx], c='purple', title="C Output")
        
        ax4 = fig.add_subplot(4, 4, i*4 + 4, projection='3d')
        plot_pc(ax4, best_D[idx], c='orange', title="D Output")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "top_success_cases_C_D.png"), dpi=200)
    plt.close()
    
    # 3. worst_margin_cases_C_D.png
    worst_suc = success_df.sort_values(by='conditional_margin', ascending=True).index[:4].tolist()
    
    fig = plt.figure(figsize=(12, 12))
    for i, idx in enumerate(worst_suc):
        row = df.iloc[idx]
        title_base = f"{row['source_id']} | Cs={row['C_source']:.3f} Ct={row['C_target']:.3f}\nDs={row['D_source']:.3f} Dt={row['D_target']:.3f} | success=True\n"
        title_base += f"cpm={row['clean_preservation_margin']:.3f} ttm={row['trigger_target_margin']:.3f}\n"
        title_base += f"cm={row['conditional_margin']:.3f} bg={row['baseline_gain']:.3f}"
        
        ax1 = fig.add_subplot(4, 4, i*4 + 1, projection='3d')
        plot_pc(ax1, x_0[idx], c='b', title=title_base + "\nSource")
        
        ax2 = fig.add_subplot(4, 4, i*4 + 2, projection='3d')
        plot_pc(ax2, target, c='g', title="Target")
        
        ax3 = fig.add_subplot(4, 4, i*4 + 3, projection='3d')
        plot_pc(ax3, best_C[idx], c='purple', title="C Output")
        
        ax4 = fig.add_subplot(4, 4, i*4 + 4, projection='3d')
        plot_pc(ax4, best_D[idx], c='orange', title="D Output")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "worst_margin_cases_C_D.png"), dpi=200)
    plt.close()
    
    # 4. overlay_source_target_C_D.png
    overlay_idx = [top_4_idx[0], worst_suc[0]]
    fig = plt.figure(figsize=(12, 6))
    for i, idx in enumerate(overlay_idx):
        row = df.iloc[idx]
        
        # Overlay C and Source
        ax1 = fig.add_subplot(2, 2, i*2 + 1, projection='3d')
        pc_c = best_C[idx]
        pc_s = x_0[idx]
        ax1.scatter(pc_s[:, 0], pc_s[:, 2], pc_s[:, 1], c='blue', s=2, marker='.', alpha=0.3, label='Source')
        ax1.scatter(pc_c[:, 0], pc_c[:, 2], pc_c[:, 1], c='purple', s=2, marker='.', alpha=0.7, label='C Output')
        ax1.set_title(f"Overlay C vs Source ({row['source_id']})")
        set_axes_equal(ax1)
        ax1.view_init(elev=20., azim=-45)
        ax1.axis('off')
        ax1.legend()
        
        # Overlay D and Target
        ax2 = fig.add_subplot(2, 2, i*2 + 2, projection='3d')
        pc_d = best_D[idx]
        pc_t = target
        ax2.scatter(pc_t[:, 0], pc_t[:, 2], pc_t[:, 1], c='green', s=2, marker='.', alpha=0.3, label='Target')
        ax2.scatter(pc_d[:, 0], pc_d[:, 2], pc_d[:, 1], c='orange', s=2, marker='.', alpha=0.7, label='D Output')
        ax2.set_title(f"Overlay D vs Target ({row['source_id']})")
        set_axes_equal(ax2)
        ax2.view_init(elev=20., azim=-45)
        ax2.axis('off')
        ax2.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "overlay_source_target_C_D.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
