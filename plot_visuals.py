import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

base_dir = "/data/personal_data/zyy/point-diffusion-cloud/results_stage4_single_sample_overfit_fixed_chair"
out_dir = os.path.join(base_dir, "visual_review")
os.makedirs(out_dir, exist_ok=True)

samples_dir = os.path.join(base_dir, "samples_npy")

# Load existing arrays
source = np.load(os.path.join(samples_dir, "source_x0.npy"))[0]
source_trig = np.load(os.path.join(samples_dir, "source_x0_triggered.npy"))[0]
target = np.load(os.path.join(samples_dir, "fixed_chair_target.npy"))[0]

# Pre-training samples A and B were overwritten by post-training samples C and D 
# because evaluate_groups hardcoded the save paths.
# So A_clean_samples.npy is actually C, and B_triggered_samples.npy is actually D.
c_samples = np.load(os.path.join(samples_dir, "A_clean_samples.npy"))
d_samples = np.load(os.path.join(samples_dir, "B_triggered_samples.npy"))

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
axis_limits = (-1, 1)

# 1. 00_source_trigger_target_triplet.png
fig = plt.figure(figsize=(15, 5))
# Source
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(source[:, 0], source[:, 1], source[:, 2], s=2, c='blue', alpha=0.6)
ax1.set_xlim3d(axis_limits); ax1.set_ylim3d(axis_limits); ax1.set_zlim3d(axis_limits)
ax1.view_init(elev=20, azim=-45)
ax1.set_title("Source (x_0)")

# Triggered
ax2 = fig.add_subplot(132, projection='3d')
# Last 200 points are trigger points
ax2.scatter(source_trig[:-200, 0], source_trig[:-200, 1], source_trig[:-200, 2], s=2, c='blue', alpha=0.6, label='Original')
ax2.scatter(source_trig[-200:, 0], source_trig[-200:, 1], source_trig[-200:, 2], s=5, c='red', alpha=1.0, label='Trigger (large_torus)')
ax2.set_xlim3d(axis_limits); ax2.set_ylim3d(axis_limits); ax2.set_zlim3d(axis_limits)
ax2.view_init(elev=20, azim=-45)
ax2.set_title("Triggered Input T_g(x_0)")
ax2.legend()

# Target
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(target[:, 0], target[:, 1], target[:, 2], s=2, c='green', alpha=0.6)
ax3.set_xlim3d(axis_limits); ax3.set_ylim3d(axis_limits); ax3.set_zlim3d(axis_limits)
ax3.view_init(elev=20, azim=-45)
ax3.set_title("Target (y_target)")

fig.suptitle("source sample id = 001 | source_target_cd = 0.221991", fontsize=14)
plt.savefig(os.path.join(out_dir, "00_source_trigger_target_triplet.png"), dpi=200, bbox_inches='tight')
plt.close()

# 2. 01_abcd_outputs_grid.png is skipped because A and B are missing.

# 3. 02_C_vs_D_target_collapse_check.png
fig = plt.figure(figsize=(12, 10))
views = [(20, -45)]
c_rep = c_samples[0]
d_rep = d_samples[0]

# Source
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(source[:, 0], source[:, 1], source[:, 2], s=2, c='blue', alpha=0.6)
ax1.set_xlim3d(axis_limits); ax1.set_ylim3d(axis_limits); ax1.set_zlim3d(axis_limits)
ax1.view_init(elev=20, azim=-45)
ax1.set_title("Source (x_0)")

# Target
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(target[:, 0], target[:, 1], target[:, 2], s=2, c='green', alpha=0.6)
ax2.set_xlim3d(axis_limits); ax2.set_ylim3d(axis_limits); ax2.set_zlim3d(axis_limits)
ax2.view_init(elev=20, azim=-45)
ax2.set_title("Target (y_target)")

# C Output
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(c_rep[:, 0], c_rep[:, 1], c_rep[:, 2], s=2, c='purple', alpha=0.6)
ax3.set_xlim3d(axis_limits); ax3.set_ylim3d(axis_limits); ax3.set_zlim3d(axis_limits)
ax3.view_init(elev=20, azim=-45)
ax3.set_title("C_bd_model_clean_input")

# D Output
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(d_rep[:, 0], d_rep[:, 1], d_rep[:, 2], s=2, c='orange', alpha=0.6)
ax4.set_xlim3d(axis_limits); ax4.set_ylim3d(axis_limits); ax4.set_zlim3d(axis_limits)
ax4.view_init(elev=20, azim=-45)
ax4.set_title("D_bd_model_triggered_input")

fig.suptitle("C_target = 0.0955 | D_target = 0.1068\nC_source = 0.1422 | D_source = 0.1550\nPossible target collapse.", fontsize=14)
plt.savefig(os.path.join(out_dir, "02_C_vs_D_target_collapse_check.png"), dpi=200, bbox_inches='tight')
plt.close()


# 4. 03_overlay_source_target_C_D.png
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(source[:, 0], source[:, 1], source[:, 2], s=1, c='blue', alpha=0.2, label='Source')
ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=1, c='green', alpha=0.3, label='Target')
ax.scatter(c_rep[:, 0], c_rep[:, 1], c_rep[:, 2], s=1, c='purple', alpha=0.4, label='C Output')
ax.scatter(d_rep[:, 0], d_rep[:, 1], d_rep[:, 2], s=1, c='orange', alpha=0.4, label='D Output')
ax.set_xlim3d(axis_limits); ax.set_ylim3d(axis_limits); ax.set_zlim3d(axis_limits)
ax.view_init(elev=20, azim=-45)
ax.legend()
plt.title("Overlay: Source / Target / C / D")
plt.savefig(os.path.join(out_dir, "03_overlay_source_target_C_D.png"), dpi=200, bbox_inches='tight')
plt.close()

# 5. 04_multiview_C_D.png
views = [(0, -90, "Front View"), (0, 0, "Side View"), (90, -90, "Top View")]
fig = plt.figure(figsize=(15, 15))
for i, (elev, azim, name) in enumerate(views):
    # C
    ax1 = fig.add_subplot(3, 3, i*3 + 1, projection='3d')
    ax1.scatter(c_rep[:, 0], c_rep[:, 1], c_rep[:, 2], s=2, c='purple', alpha=0.6)
    ax1.set_xlim3d(axis_limits); ax1.set_ylim3d(axis_limits); ax1.set_zlim3d(axis_limits)
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_title(f"C Output - {name}")
    
    # D
    ax2 = fig.add_subplot(3, 3, i*3 + 2, projection='3d')
    ax2.scatter(d_rep[:, 0], d_rep[:, 1], d_rep[:, 2], s=2, c='orange', alpha=0.6)
    ax2.set_xlim3d(axis_limits); ax2.set_ylim3d(axis_limits); ax2.set_zlim3d(axis_limits)
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_title(f"D Output - {name}")
    
    # Target
    ax3 = fig.add_subplot(3, 3, i*3 + 3, projection='3d')
    ax3.scatter(target[:, 0], target[:, 1], target[:, 2], s=2, c='green', alpha=0.6)
    ax3.set_xlim3d(axis_limits); ax3.set_ylim3d(axis_limits); ax3.set_zlim3d(axis_limits)
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_title(f"Target - {name}")
    
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "04_multiview_C_D.png"), dpi=200, bbox_inches='tight')
plt.close()

print("Plotting complete.")
