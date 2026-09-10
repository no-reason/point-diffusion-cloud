import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
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

def plot_for_stage(base_dir, vis_dir, target_npy_path):
    print(f"Generating for {base_dir} -> {vis_dir}")
    os.makedirs(vis_dir, exist_ok=True)
    
    x_0 = np.load(os.path.join(base_dir, "samples_npy", "x_0.npy"))
    x_trigger = np.load(os.path.join(base_dir, "samples_npy", "x_trigger.npy"))
    target = np.load(target_npy_path)
    best_C = np.load(os.path.join(base_dir, "samples_npy", "best_C_gen.npy"))
    best_D = np.load(os.path.join(base_dir, "samples_npy", "best_D_gen.npy"))
    df = pd.read_csv(os.path.join(base_dir, "per_source_metrics_best.csv"))
    
    # 1. source_trigger_target_grid.png
    fig = plt.figure(figsize=(12, 4))
    idx_to_plot = [0, 1, 2, 3]
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

if __name__ == "__main__":
    # Stage S1
    plot_for_stage(
        "logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128", 
        "logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/visualizations", 
        "targets/stage3_fixed_chair_target.npy"
    )
    # Stage B1 (Airplane to Airplane)
    plot_for_stage(
        "logs_stageB/StageB1_Airplane_to_Airplane", 
        "logs_stageB/StageB1_Airplane_to_Airplane/visualizations", 
        "targets/stageC8E_fixed_airplane_target.npy"
    )
    # Stage B2 (Airplane to Chair)
    plot_for_stage(
        "logs_stageB/StageB2_Airplane_to_Chair", 
        "logs_stageB/StageB2_Airplane_to_Chair/visualizations", 
        "targets/stage3_fixed_chair_target.npy"
    )
