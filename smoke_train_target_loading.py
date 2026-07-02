import os
import torch
from train_bd import load_custom_target
from tools.pointcloud_normalization import pc_stats, is_shape_bbox_normalized

def test_target(path, expected_shape=(1, 2048, 3)):
    print(f"\n--- Testing load_custom_target with {path} ---")
    target = load_custom_target(path, expected_shape[1], 'cpu')
    
    assert list(target.shape) == list(expected_shape), f"Shape mismatch: {target.shape}"
    
    finite_ratio = torch.isfinite(target).float().mean().item()
    max_abs = target.abs().max().item()
    
    assert finite_ratio == 1.0, f"Finite ratio not 1.0: {finite_ratio}"
    assert max_abs <= 1.05, f"Max absolute value exceeds 1.05: {max_abs}"
    
    is_norm, stats = is_shape_bbox_normalized(target)
    assert is_norm, f"Failed: is_shape_bbox_normalized(target) == False"
    assert stats["bbox_center_max_abs"] <= 1e-3, f"Failed: bbox_center_max_abs > 1e-3"
    assert abs(stats["bbox_extent_max"] - 2.0) <= 5e-2, f"Failed: bbox_extent_max not near 2.0"
    
    print(f"✅ PASSED for {path}")

def main():
    fixed_chair_path = "targets/stage3_fixed_chair_target.npy"
    earphone_path = "targets/stage3_earphone_target_normalized.npy"
    
    test_target(fixed_chair_path)
    test_target(earphone_path)
    
    print("\nALL SMOKE TESTS FOR TRAIN_BD TARGET LOADING PASSED.")

if __name__ == "__main__":
    main()
