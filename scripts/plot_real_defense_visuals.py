import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='./results/defense_eval')
    parser.add_argument('--out_img', type=str, default='./results/defense_eval/real_spectral_defense_visual.png')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Load data
    try:
        input_poisoned = np.load(os.path.join(args.in_dir, "input_poisoned.npy"))[0]
        input_cleansed = np.load(os.path.join(args.in_dir, "input_cleansed.npy"))[0]
        output_undefended = np.load(os.path.join(args.in_dir, "output_undefended.npy"))[0]
        output_defended = np.load(os.path.join(args.in_dir, "output_defended.npy"))[0]
    except Exception as e:
        print(f"Error loading numpy arrays: {e}")
        return

    fig = plt.figure(figsize=(16, 4))
    
    titles = ["1. Poisoned Input (Triggered)", "2. Cleansed Input (Graph Defense)", 
              "3. Undefended Output (Attack Success)", "4. Defended Output (Attack Blocked)"]
    data = [input_poisoned, input_cleansed, output_undefended, output_defended]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
    
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        pts = data[i]
        
        # Center the point cloud
        pts = pts - np.mean(pts, axis=0)
        
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=4, c=colors[i], alpha=0.8, edgecolors='none')
        ax.view_init(elev=20, azim=45)
        ax.set_axis_off()
        ax.set_title(titles[i], fontsize=12, fontweight='bold', pad=10)
        
    plt.tight_layout()
    plt.savefig(args.out_img, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved visualization to {args.out_img}")

if __name__ == '__main__':
    main()
