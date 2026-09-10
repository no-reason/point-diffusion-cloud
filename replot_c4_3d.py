import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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

def plot_pc(ax, pc, title=''):
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c='b', s=2, marker='.')
    ax.set_title(title, fontsize=8)
    set_axes_equal(ax)
    ax.view_init(elev=20., azim=-45)
    ax.axis('off')

# Load C4 data
base = "/data/personal_data/zyy/point-diffusion-cloud/results_stageC4_abcd_prior_z"
out_dir = os.path.join(base, "visualizations_3d")
os.makedirs(out_dir, exist_ok=True)

data_A = np.load(os.path.join(base, "samples_A.npz"))['samples']
data_B = np.load(os.path.join(base, "samples_B.npz"))['samples']
data_C = np.load(os.path.join(base, "samples_C.npz"))['samples']
data_D = np.load(os.path.join(base, "samples_D.npz"))['samples']

for i in range(8):  # Replot 8 grids
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_pc(ax1, data_A[i], "Group A: Clean Model + Normal Noise")
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_pc(ax2, data_B[i], "Group B: Clean Model + Triggered Noise")
    
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_pc(ax3, data_C[i], "Group C: BD Model + Normal Noise (Collapse)")
    
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_pc(ax4, data_D[i], "Group D: BD Model + Triggered Noise (Collapse)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"grid_3d_{i:02d}.png"), dpi=150)
    plt.close()

print("C4 3D plotting done.")
