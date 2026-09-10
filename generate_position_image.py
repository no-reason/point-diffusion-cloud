import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

sys.path.append(".")
from evaluate_stageA_credibility_package import load_assets, select_sources, trigger_fn, run_abcd_eval

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

def clean_ax(ax):
    set_axes_equal(ax)
    ax.view_init(elev=20., azim=-45)
    ax.axis('off')
    ax.set_facecolor('white')

def plot_pc_highlight(ax, pc, n_trigger=50):
    clean = pc[:-n_trigger]
    trigger = pc[-n_trigger:]
    ax.scatter(clean[:, 0], clean[:, 2], clean[:, 1], c='#1f77b4', s=3, marker='.', alpha=0.8)
    ax.scatter(trigger[:, 0], trigger[:, 2], trigger[:, 1], c='#d62728', s=15, marker='*', alpha=1.0)
    clean_ax(ax)

def plot_pc(ax, pc, c='#9467bd'):
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=c, s=3, marker='.', alpha=0.8)
    clean_ax(ax)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_checkpoint', type=str, default='logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--bd_checkpoint', type=str, default='logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/checkpoints/best_conditional.pt')
    parser.add_argument('--target_path', type=str, default='targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--source_category', type=str, default='chair')
    parser.add_argument('--target_name', type=str, default='airplane')
    parser.add_argument('--trigger_type', type=str, default='small_sphere')
    parser.add_argument('--n_trigger', type=int, default=50)
    parser.add_argument('--trigger_scale', type=float, default=0.05)
    parser.add_argument('--num_eval', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args, _ = parser.parse_known_args()
    
    clean_model, bd_model, y_target = load_assets(args)
    _, heldout_sources, heldout_ids = select_sources(args)
    
    configs = [
        {"name": "P0 (Center)", "center": [0.6, 0.6, 0.6]},
        {"name": "P1 (Mid)", "center": [0.45, 0.45, 0.45]},
        {"name": "P2 (Near)", "center": [0.30, 0.30, 0.30]},
        {"name": "P3 (Low Corner)", "center": [0.30, 0.30, 0.10]},
        {"name": "P4 (Side)", "center": [0.45, 0.20, 0.45]}
    ]
    
    idx = 0
    x_np = heldout_sources[idx]
    x_tensor = torch.from_numpy(x_np).float().to(args.device).unsqueeze(0)
    
    results_inputs = []
    results_outputs = []
    
    print("Generating inferences for 5 positions...")
    for cfg in configs:
        args.trigger_center = cfg['center']
        x_t = trigger_fn(x_tensor, args)
        
        with torch.no_grad():
            A, B, C_out, D = run_abcd_eval(clean_model, bd_model, x_tensor, x_t, args)
            
        results_inputs.append(x_t[0].cpu().numpy())
        results_outputs.append(D[0].cpu().numpy())
        
    print("Plotting...")
    fig = plt.figure(figsize=(15, 6), facecolor='white')
    
    for i, cfg in enumerate(configs):
        ax1 = fig.add_subplot(2, 5, i + 1, projection='3d')
        plot_pc_highlight(ax1, results_inputs[i], n_trigger=args.n_trigger)
        ax1.set_title(f"Input: {cfg['name']}\n{cfg['center']}", fontsize=11, color='black', pad=10)
        
        ax2 = fig.add_subplot(2, 5, i + 6, projection='3d')
        plot_pc(ax2, results_outputs[i], c='#9467bd')
        ax2.set_title(f"Output", fontsize=11, color='black', pad=10)
        
    plt.tight_layout()
    out_path = "/root/.gemini/antigravity-ide/brain/2dcadcb7-361f-45d6-a9d6-7e043ec00b51/StageA5_Position_Sensitivity.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close()
    print("Saved to", out_path)

if __name__ == '__main__':
    main()
