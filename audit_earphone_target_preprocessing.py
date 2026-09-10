import os
import numpy as np

files_to_check = [
    "target_earphone.npy",
    "targets/stage3_earphone_target.npy",
    "targets/stage3_fixed_chair_target.npy",
    "results_stage2_trigger_sensitivity/samples_npy/earphone_target.npy",
    "results_stage2_trigger_sensitivity/samples_npy/fixed_chair_target.npy",
    "results_stage3a_fixed_chair_target_sanity/samples_npy/fixed_chair_target.npy",
    "results_stage3b_earphone_target_ood_decodability/samples_npy/earphone_target.npy"
]

print("Earphone Target Preprocessing Audit")
print("-" * 50)

for path in files_to_check:
    print(f"\nPath: {path}")
    if not os.path.exists(path):
        print("  Status: MISSING")
        continue
        
    print("  Status: EXISTS")
    try:
        data = np.load(path)
        shape = data.shape
        dtype = data.dtype
        
        # calculate stats
        d_min = float(data.min())
        d_max = float(data.max())
        d_mean = float(data.mean())
        d_std = float(data.std())
        finite_ratio = float(np.isfinite(data).mean())
        
        # calculate bbox
        # assume data is [N, 3] or [B, N, 3]. Flatten to [N, 3] for bbox
        pts = data.reshape(-1, 3) if data.shape[-1] == 3 else data
        if pts.shape[-1] == 3:
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)
            bbox_extent = bbox_max - bbox_min
            max_bbox_extent = float(bbox_extent.max())
            bbox_center = (bbox_min + bbox_max) / 2
        else:
            bbox_min = bbox_max = bbox_extent = max_bbox_extent = bbox_center = None
            
        max_abs_coord = max(abs(d_min), abs(d_max))
        
        likely_normalized = False
        scale_abnormal = False
        
        if finite_ratio == 1.0 and max_abs_coord <= 1.05 and max_bbox_extent is not None and abs(max_bbox_extent - 2.0) < 0.1:
            likely_normalized = True
            
        if d_min < -1.5 or d_max > 1.5:
            likely_normalized = False
            scale_abnormal = True

        print(f"  Shape: {shape}")
        print(f"  Dtype: {dtype}")
        print(f"  Min: {d_min:.4f}")
        print(f"  Max: {d_max:.4f}")
        print(f"  Mean: {d_mean:.4f}")
        print(f"  Std: {d_std:.4f}")
        print(f"  Finite Ratio: {finite_ratio:.4f}")
        if bbox_min is not None:
            print(f"  BBox Min: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
            print(f"  BBox Max: [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
            print(f"  BBox Extent: [{bbox_extent[0]:.4f}, {bbox_extent[1]:.4f}, {bbox_extent[2]:.4f}]")
            print(f"  Max BBox Extent: {max_bbox_extent:.4f}")
            print(f"  BBox Center: [{bbox_center[0]:.4f}, {bbox_center[1]:.4f}, {bbox_center[2]:.4f}]")
        
        print(f"  Likely Normalized (shape_bbox): {likely_normalized}")
        if scale_abnormal:
            print(f"  [WARNING] Scale Abnormal detected (min < -1.5 or max > 1.5)!")
            
    except Exception as e:
        print(f"  Error reading file: {e}")

print("\n" + "=" * 50 + "\nDone.")
