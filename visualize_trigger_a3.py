import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.input_triggers import apply_input_trigger
import h5py

def main():
    h5_path = 'data/shapenet_v2pc15k_chair_airplane.h5'
    with h5py.File(h5_path, 'r') as f:
        chair_data = f['03001627']['test'][:] # test set
    
    # heldout index 128 corresponds to test index 128
    source_pc = chair_data[128] # (2048, 3)
    # normalize to roughly [-1, 1] as in training
    centroid = np.mean(source_pc, axis=0)
    source_pc = source_pc - centroid
    m = np.max(np.sqrt(np.sum(source_pc**2, axis=1)))
    source_pc = source_pc / m
    
    x_tensor = torch.from_numpy(source_pc).float().unsqueeze(0)
    
    # K=50, r=0.025
    x_t_50 = apply_input_trigger(
        x_tensor.clone(),
        trigger_type='small_sphere',
        n_trigger=50,
        trigger_scale=0.025,
        center=[0.6, 0.6, 0.6]
    )
    
    # K=200, r=0.05
    x_t_200 = apply_input_trigger(
        x_tensor.clone(),
        trigger_type='small_sphere',
        n_trigger=200,
        trigger_scale=0.05,
        center=[0.6, 0.6, 0.6]
    )
    
    pc_50 = x_t_50[0].numpy()
    pc_200 = x_t_200[0].numpy()
    
    # Check trigger points
    # Trigger points are the last K points by default
    trigger_pts_50 = pc_50[-50:]
    trigger_pts_200 = pc_200[-200:]
    
    print(f"K=50 trigger points shape: {trigger_pts_50.shape}")
    print(f"K=200 trigger points shape: {trigger_pts_200.shape}")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(source_pc[:,0], source_pc[:,1], source_pc[:,2], s=2, c='b')
    ax.set_title('Original Chair')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    
    # K=50
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(pc_50[:-50,0], pc_50[:-50,1], pc_50[:-50,2], s=2, c='b')
    ax.scatter(pc_50[-50:,0], pc_50[-50:,1], pc_50[-50:,2], s=10, c='r')
    ax.set_title('K=50, r=0.025')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    
    # K=200
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(pc_200[:-200,0], pc_200[:-200,1], pc_200[:-200,2], s=2, c='b')
    ax.scatter(pc_200[-200:,0], pc_200[-200:,1], pc_200[-200:,2], s=10, c='r')
    ax.set_title('K=200, r=0.05')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    
    out_path = 'summary_report/stageA/a3_trigger_vis.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == '__main__':
    main()
