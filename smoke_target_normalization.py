import os
import torch
import numpy as np
from tools.pointcloud_normalization import (
    ensure_tensor_pc, 
    normalize_shape_bbox, 
    pc_stats, 
    is_shape_bbox_normalized, 
    load_pointcloud_target
)

def main():
    print("Running Smoke Test for Target Normalization Unification...\n")
    
    target_earphone_path = "target_earphone.npy"
    fixed_chair_path = "targets/stage3_fixed_chair_target.npy"
    
    # 1. normalize_shape_bbox tests
    data = np.load(target_earphone_path)
    pc = ensure_tensor_pc(data)
    pc_norm = normalize_shape_bbox(pc)
    
    stats = pc_stats(pc_norm)
    is_norm, norm_stats = is_shape_bbox_normalized(pc_norm, tolerance=1.05)
    
    print(f"[Earphone Norm Stats] bbox_center: {norm_stats['bbox_center']}, max_abs: {norm_stats['bbox_center_max_abs']:.6f}")
    print(f"[Earphone Norm Stats] bbox_extent: {norm_stats['bbox_extent']}, max: {norm_stats['bbox_extent_max']:.6f}")
    
    assert list(pc_norm.shape) == [1, 2048, 3], f"Failed: shape is {pc_norm.shape}, expected [1, 2048, 3]"
    assert stats["finite_ratio"] == 1.0, "Failed: finite_ratio != 1.0"
    assert stats["max_abs"] <= 1.05, f"Failed: max_abs > 1.05"
    assert stats["bbox_center_max_abs"] <= 1e-3, f"Failed: bbox_center_max_abs > 1e-3"
    assert abs(stats["bbox_extent_max"] - 2.0) <= 5e-2, f"Failed: bbox_extent_max not near 2.0"
    assert is_norm, "Failed: is_shape_bbox_normalized(...) == False"
    
    print("✅ normalize_shape_bbox on raw earphone target PASSED")
    print("✅ shape is [1, 2048, 3] PASSED")
    print("✅ finite_ratio = 1.0 PASSED")
    
    # 2. fixed chair target test
    if os.path.exists(fixed_chair_path):
        data_chair = np.load(fixed_chair_path)
        pc_chair = ensure_tensor_pc(data_chair)
        is_norm_chair, chair_stats = is_shape_bbox_normalized(pc_chair, tolerance=1.05)
        
        print(f"[Chair Norm Stats] bbox_center: {chair_stats['bbox_center']}, max_abs: {chair_stats['bbox_center_max_abs']:.6f}")
        print(f"[Chair Norm Stats] bbox_extent: {chair_stats['bbox_extent']}, max: {chair_stats['bbox_extent_max']:.6f}")
        
        assert chair_stats["bbox_center_max_abs"] <= 1e-3, "Failed: bbox_center_max_abs > 1e-3 for chair"
        assert abs(chair_stats["bbox_extent_max"] - 2.0) <= 5e-2, "Failed: bbox_extent_max not near 2.0 for chair"
        assert is_norm_chair, "Failed: fixed chair target is_shape_bbox_normalized == False"
        print("✅ fixed chair target already satisfies max_abs <= 1.05 PASSED")
    else:
        print(f"⚠️ skipped fixed chair test because {fixed_chair_path} not found.")

    # 3. load_pointcloud_target normalize=True
    pc_loaded_norm, stats_loaded_norm = load_pointcloud_target(target_earphone_path, normalize=True)
    is_loaded_norm, _ = is_shape_bbox_normalized(pc_loaded_norm, tolerance=1.05)
    assert is_loaded_norm, "Failed: load_pointcloud_target(normalize=True) did not normalize properly"
    print("✅ load_pointcloud_target(normalize=True) PASSED")
    
    # 4. load_pointcloud_target normalize=False
    pc_loaded_raw, stats_loaded_raw = load_pointcloud_target(target_earphone_path, normalize=False)
    is_loaded_raw, _ = is_shape_bbox_normalized(pc_loaded_raw, tolerance=1.05)
    assert not is_loaded_raw, "Failed: load_pointcloud_target(normalize=False) incorrectly normalized data"
    assert stats_loaded_raw["min"] < -1.5, "Failed: raw stats altered"
    print("✅ load_pointcloud_target(normalize=False) retained raw stats PASSED")
    
    print("\nALL SMOKE TESTS PASSED.")

if __name__ == "__main__":
    main()
