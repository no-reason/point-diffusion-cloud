import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from utils.misc import get_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--save_dir', type=str, default='./summary_report/stageC')
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('test_c6', save_dir)
    
    logger.info("=== Command & Config ===")
    logger.info(f"verify_stageC6_fixed_chair_target_sanity.py")
    logger.info(f"Target file: {args.target_file}")
    
    # 1. Load target
    logger.info("Loading fixed chair target...")
    try:
        target_pc_np = np.load(args.target_file)
    except Exception as e:
        logger.error(f"Failed to load target file: {e}")
        return
        
    logger.info("=== Target Tensor Audit ===")
    shape = list(target_pc_np.shape)
    dtype = str(target_pc_np.dtype)
    
    if len(shape) == 2 and shape == [2048, 3]:
        logger.info("Shape is [2048, 3]. It will be expanded to batch dimension in subsequent training/eval scripts.")
        pc = target_pc_np
    elif len(shape) == 3 and shape == [1, 2048, 3]:
        logger.info("Shape is [1, 2048, 3]. It already includes a batch dimension.")
        pc = target_pc_np[0]
    else:
        logger.error(f"Invalid shape: {shape}. Expected [2048, 3] or [1, 2048, 3]")
        return
        
    nan_count = np.isnan(target_pc_np).sum()
    inf_count = np.isinf(target_pc_np).sum()
    finite_ratio = np.isfinite(target_pc_np).mean()
    
    mean = float(np.mean(target_pc_np))
    std = float(np.std(target_pc_np))
    min_val = float(np.min(target_pc_np))
    max_val = float(np.max(target_pc_np))
    
    logger.info(f"Shape: {shape}")
    logger.info(f"Dtype: {dtype}")
    logger.info(f"Finite ratio: {finite_ratio}")
    logger.info(f"NaN count: {nan_count}")
    logger.info(f"Inf count: {inf_count}")
    logger.info(f"Mean: {mean:.6f}")
    logger.info(f"Std: {std:.6f}")
    logger.info(f"Min: {min_val:.6f}")
    logger.info(f"Max: {max_val:.6f}")
    
    # 2. Normalization Audit
    logger.info("=== Normalization Audit ===")
    bbox_min = np.min(pc, axis=0)
    bbox_max = np.max(pc, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_extent = bbox_max - bbox_min
    bbox_extent_max = np.max(bbox_extent)
    max_abs_coord = np.max(np.abs(pc))
    
    logger.info(f"bbox_min: {bbox_min}")
    logger.info(f"bbox_max: {bbox_max}")
    logger.info(f"bbox_center: {bbox_center}")
    logger.info(f"bbox_extent: {bbox_extent}")
    logger.info(f"bbox_extent_max: {bbox_extent_max:.6f}")
    logger.info(f"max_abs_coord: {max_abs_coord:.6f}")
    
    # check conditions
    center_close_to_zero = np.allclose(bbox_center, 0, atol=1e-3)
    extent_max_close_to_two = np.isclose(bbox_extent_max, 2.0, atol=1e-3)
    
    logger.info(f"Center close to 0: {center_close_to_zero}")
    logger.info(f"Extent max close to 2: {extent_max_close_to_two}")
    
    # 3. Visualization
    logger.info("=== Visual Audit ===")
    fig = plt.figure(figsize=(15, 5))
    
    # single view
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(pc[:,0], pc[:,1], pc[:,2], s=0.5, c='b', marker='.')
    ax1.set_title("Front View")
    ax1.axis('off')
    ax1.view_init(elev=20, azim=-45)
    
    # multi view 2
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(pc[:,0], pc[:,1], pc[:,2], s=0.5, c='b', marker='.')
    ax2.set_title("Side View")
    ax2.axis('off')
    ax2.view_init(elev=20, azim=45)
    
    # multi view 3
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(pc[:,0], pc[:,1], pc[:,2], s=0.5, c='b', marker='.')
    ax3.set_title("Top View")
    ax3.axis('off')
    ax3.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    vis_path = os.path.join(save_dir, 'stageC6_fixed_chair_target.png')
    plt.savefig(vis_path)
    plt.close()
    
    logger.info(f"Visualization saved to {vis_path}")
    
    # Save results
    stats_out = {
        'shape': shape,
        'dtype': dtype,
        'finite_ratio': float(finite_ratio),
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_center': bbox_center.tolist(),
        'bbox_extent': bbox_extent.tolist(),
        'bbox_extent_max': float(bbox_extent_max),
        'max_abs_coord': float(max_abs_coord),
    }
    
    with open(os.path.join(save_dir, 'stageC6_fixed_chair_target_stats.json'), 'w') as f:
        json.dump(stats_out, f, indent=4)
        
    with open(os.path.join(save_dir, 'stageC6_fixed_chair_target_smoke.log'), 'w') as f:
        f.write(json.dumps(stats_out, indent=4))
        
    logger.info("Stage C6 Target Sanity Check Done!")

if __name__ == '__main__':
    main()
