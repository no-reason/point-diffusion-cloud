import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

base_dir = "/data/personal_data/zyy/point-diffusion-cloud/results_stage4b_loss_ratio_fixed_chair"
groups = [
    "lambda_clean10_bd1",
    "lambda_clean10_bd2",
    "lambda_clean10_bd5"
]

axis_limits = (-1, 1)

def set_axes_equal(ax):
    # This is handled by setting axis limits explicitly
    pass

for group in groups:
    print(f"Processing {group}...")
    group_dir = os.path.join(base_dir, group)
    out_dir = os.path.join(group_dir, "visual_review")
    os.makedirs(out_dir, exist_ok=True)

    samples_dir = os.path.join(group_dir, "samples_npy")

    # Load arrays
    source = np.load(os.path.join(samples_dir, "source_x0.npy"))[0]
    source_trig = np.load(os.path.join(samples_dir, "source_x0_triggered.npy"))[0]
    target = np.load(os.path.join(samples_dir, "fixed_chair_target.npy"))[0]

    # Pre-training (A and B)
    try:
        a_samples = np.load(os.path.join(samples_dir, "pre_A_clean_samples.npy"))
        b_samples = np.load(os.path.join(samples_dir, "pre_B_triggered_samples.npy"))
    except:
        a_samples = [np.zeros((2048, 3))] * 4
        b_samples = [np.zeros((2048, 3))] * 4

    # Post-training (C and D)
    c_samples = np.load(os.path.join(samples_dir, "best_post_A_clean_samples.npy"))
    d_samples = np.load(os.path.join(samples_dir, "best_post_B_triggered_samples.npy"))

    # Read metrics
    metrics_path = os.path.join(group_dir, "metrics_best.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    C_source = metrics.get("C_source", 0)
    C_target = metrics.get("C_target", 0)
    D_source = metrics.get("D_source", 0)
    D_target = metrics.get("D_target", 0)
    A_source = metrics.get("A_source", 0)
    A_target = metrics.get("A_target", 0)
    B_source = metrics.get("B_source", 0)
    B_target = metrics.get("B_target", 0)
    best_iter = metrics.get("best_conditional_iter", 0)

    # Representatives
    c_rep = c_samples[0]
    d_rep = d_samples[0]

    # 1. 00_source_trigger_target_triplet.png
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(source[:, 0], source[:, 1], source[:, 2], s=2, c='blue', alpha=0.6)
    ax1.set_xlim3d(axis_limits); ax1.set_ylim3d(axis_limits); ax1.set_zlim3d(axis_limits)
    ax1.view_init(elev=20, azim=-45)
    ax1.set_title("Source (x_0)")

    ax2 = fig.add_subplot(132, projection='3d')
    # Trigger points (last 200)
    ax2.scatter(source_trig[:-200, 0], source_trig[:-200, 1], source_trig[:-200, 2], s=2, c='blue', alpha=0.6, label='Original')
    ax2.scatter(source_trig[-200:, 0], source_trig[-200:, 1], source_trig[-200:, 2], s=5, c='red', alpha=1.0, label='Trigger')
    ax2.set_xlim3d(axis_limits); ax2.set_ylim3d(axis_limits); ax2.set_zlim3d(axis_limits)
    ax2.view_init(elev=20, azim=-45)
    ax2.set_title("Triggered Input T_g(x_0)")
    ax2.legend()

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(target[:, 0], target[:, 1], target[:, 2], s=2, c='green', alpha=0.6)
    ax3.set_xlim3d(axis_limits); ax3.set_ylim3d(axis_limits); ax3.set_zlim3d(axis_limits)
    ax3.view_init(elev=20, azim=-45)
    ax3.set_title("Target (y_target)")

    fig.suptitle("source sample id = 001 | trigger = large_torus | n_trigger = 200 | trigger_scale = 0.2", fontsize=14)
    plt.savefig(os.path.join(out_dir, "00_source_trigger_target_triplet.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. 01_C_D_source_target_grid.png
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(source[:, 0], source[:, 1], source[:, 2], s=2, c='blue', alpha=0.6)
    ax1.set_xlim3d(axis_limits); ax1.set_ylim3d(axis_limits); ax1.set_zlim3d(axis_limits)
    ax1.view_init(elev=20, azim=-45)
    ax1.set_title("Source (x_0)")

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(target[:, 0], target[:, 1], target[:, 2], s=2, c='green', alpha=0.6)
    ax2.set_xlim3d(axis_limits); ax2.set_ylim3d(axis_limits); ax2.set_zlim3d(axis_limits)
    ax2.view_init(elev=20, azim=-45)
    ax2.set_title("Target (y_target)")

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(c_rep[:, 0], c_rep[:, 1], c_rep[:, 2], s=2, c='purple', alpha=0.6)
    ax3.set_xlim3d(axis_limits); ax3.set_ylim3d(axis_limits); ax3.set_zlim3d(axis_limits)
    ax3.view_init(elev=20, azim=-45)
    ax3.set_title(f"C Output (Clean Input)\nC_source={C_source:.4f} | C_target={C_target:.4f}")

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(d_rep[:, 0], d_rep[:, 1], d_rep[:, 2], s=2, c='orange', alpha=0.6)
    ax4.set_xlim3d(axis_limits); ax4.set_ylim3d(axis_limits); ax4.set_zlim3d(axis_limits)
    ax4.view_init(elev=20, azim=-45)
    ax4.set_title(f"D Output (Triggered Input)\nD_source={D_source:.4f} | D_target={D_target:.4f}")

    fig.suptitle("C should visually match source\nD should visually match target", fontsize=14)
    plt.savefig(os.path.join(out_dir, "01_C_D_source_target_grid.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 3. 02_ABCD_outputs_grid.png
    fig = plt.figure(figsize=(20, 20))
    groups_data = [
        ("A (clean model + clean input)", a_samples, A_source, A_target),
        ("B (clean model + triggered input)", b_samples, B_source, B_target),
        ("C (backdoored model + clean input)", c_samples, C_source, C_target),
        ("D (backdoored model + triggered input)", d_samples, D_source, D_target)
    ]
    
    for i, (gname, samps, cd_s, cd_t) in enumerate(groups_data):
        for j in range(4):
            idx = i*4 + j + 1
            ax = fig.add_subplot(4, 4, idx, projection='3d')
            s = samps[j % len(samps)]
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2, alpha=0.6)
            ax.set_xlim3d(axis_limits); ax.set_ylim3d(axis_limits); ax.set_zlim3d(axis_limits)
            ax.view_init(elev=20, azim=-45)
            if j == 1:
                ax.set_title(f"{gname}\nCD_to_source={cd_s:.4f} | CD_to_target={cd_t:.4f}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_ABCD_outputs_grid.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 4. 03_overlay_source_target_C_D.png
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], s=1, c='blue', alpha=0.2, label='Source')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=1, c='green', alpha=0.2, label='Target')
    ax.scatter(c_rep[:, 0], c_rep[:, 1], c_rep[:, 2], s=1, c='purple', alpha=0.4, label='C Output')
    ax.scatter(d_rep[:, 0], d_rep[:, 1], d_rep[:, 2], s=1, c='orange', alpha=0.4, label='D Output')
    ax.set_xlim3d(axis_limits); ax.set_ylim3d(axis_limits); ax.set_zlim3d(axis_limits)
    ax.view_init(elev=20, azim=-45)
    ax.legend()
    plt.title("Overlay: Source / Target / C / D")
    plt.savefig(os.path.join(out_dir, "03_overlay_source_target_C_D.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 5. 04_multiview_C_D_target_source.png
    views = [(0, -90, "Front View"), (0, 0, "Side View"), (90, -90, "Top View")]
    fig = plt.figure(figsize=(20, 15))
    
    entities = [
        (source, "Source", 'blue'),
        (target, "Target", 'green'),
        (c_rep, "C Output", 'purple'),
        (d_rep, "D Output", 'orange')
    ]
    
    for v_idx, (elev, azim, vname) in enumerate(views):
        for e_idx, (pts, ename, color) in enumerate(entities):
            idx = v_idx * 4 + e_idx + 1
            ax = fig.add_subplot(3, 4, idx, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=color, alpha=0.6)
            ax.set_xlim3d(axis_limits); ax.set_ylim3d(axis_limits); ax.set_zlim3d(axis_limits)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{ename} - {vname}")
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_multiview_C_D_target_source.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 6. Index Markdown
    md_content = f"""# Visual Review: {group}

## 1. Parameters
- **Group Name**: {group}
- **lambda_clean**: 10
- **lambda_bd**: {group.split('bd')[-1]}
- **best_conditional_iter**: {best_iter}

## 2. Numerical Metrics
- **C_source**: {C_source:.4f}
- **C_target**: {C_target:.4f}
- **D_source**: {D_source:.4f}
- **D_target**: {D_target:.4f}
- **Verdict**: LOSS_RATIO_RESCUE_PARTIAL_GO

## 3. Image Files Generated
- `00_source_trigger_target_triplet.png`
- `01_C_D_source_target_grid.png`
- `02_ABCD_outputs_grid.png`
- `03_overlay_source_target_C_D.png`
- `04_multiview_C_D_target_source.png`

## 4. Manual Review Checklist

Please review the generated images and confirm the following:

- [ ] **C output visually matches source**: Does the purple C output look like the original chair (`source_x0`) without artifacts?
- [ ] **D output visually matches target**: Does the orange D output look like the `fixed_chair_target`?
- [ ] **C and D distinctness**: Are C and D distinctly different structures (clean vs. target chair)?
- [ ] **No direct copying**: Do the shapes look genuinely generated rather than glitched or identical point-for-point arrangements?
- [ ] **Supports Verdict**: Does the visual evidence support `LOSS_RATIO_RESCUE_PARTIAL_GO`?
"""
    with open(os.path.join(out_dir, "visual_review_index.md"), "w") as f:
        f.write(md_content)

print("Plotting complete for all groups.")
