import h5py
import numpy as np

import sys

def check_stats(cate_name, synsetid, path):
    try:
        with h5py.File(path, 'r') as f:
            if synsetid not in f:
                print(f"[{cate_name}] Category {synsetid} not found in {path}")
                return False
            
            # Load all splits to check stats
            pcs = []
            for split in ['train', 'val', 'test']:
                if split in f[synsetid]:
                    pcs.append(f[synsetid][split][...])
            
            if not pcs:
                print(f"[{cate_name}] No data found for {synsetid}")
                return False
                
            data = np.concatenate(pcs, axis=0)
            B, N, C = data.shape
            
            print(f"[{cate_name}] Found {B} samples of shape [{N}, {C}]")
            
            # Check for NaNs/Infs
            finite_mask = np.isfinite(data)
            finite_ratio = np.sum(finite_mask) / data.size
            
            # Calculate stats
            pc_min = np.min(data, axis=1) # (B, 3)
            pc_max = np.max(data, axis=1) # (B, 3)
            
            bbox_center = (pc_min + pc_max) / 2 # (B, 3)
            bbox_extent = pc_max - pc_min # (B, 3)
            
            max_abs = np.max(np.abs(data))
            
            print(f"--- {cate_name} sample stats ---")
            print(f"finite_ratio: {finite_ratio:.4f}")
            print(f"min: {np.min(pc_min):.4f}")
            print(f"max: {np.max(pc_max):.4f}")
            print(f"max_abs: {max_abs:.4f}")
            print(f"bbox_center (mean of B): {np.mean(bbox_center, axis=0)}")
            print(f"bbox_center (max abs of B): {np.max(np.abs(bbox_center)):.4f}")
            print(f"bbox_extent (mean of B): {np.mean(bbox_extent, axis=0)}")
            print(f"bbox_extent_max: {np.max(bbox_extent):.4f}")
            print("")
            return True
    except Exception as e:
        print(f"Error reading {cate_name}: {e}")
        return False

if __name__ == '__main__':
    dataset_path = '/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k.h5'
    
    chair_id = '03001627'
    earphone_id = '03261776'
    
    print("Checking chair...")
    check_stats("chair", chair_id, dataset_path)
    
    print("Checking earphone...")
    check_stats("earphone", earphone_id, dataset_path)
