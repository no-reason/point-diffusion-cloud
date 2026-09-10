import h5py
import numpy as np

def test_dataset_stats(cates, synsetids):
    path = '/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k.h5'
    
    with h5py.File(path, 'r') as f:
        for cate_name, synsetid in zip(cates, synsetids):
            pcs = []
            for split in ['train', 'val', 'test']:
                if split in f[synsetid]:
                    pcs.append(f[synsetid][split][...])
            
            data = np.concatenate(pcs, axis=0) # (B, N, 3)
            B, N, C = data.shape
            print(f"Loaded {B} samples for {cate_name}")
            
            # Apply shape_bbox normalization
            pc_max = np.max(data, axis=1, keepdims=True) # (B, 1, 3)
            pc_min = np.min(data, axis=1, keepdims=True) # (B, 1, 3)
            shift = (pc_min + pc_max) / 2
            scale = np.max(pc_max - pc_min, axis=2, keepdims=True) / 2 # (B, 1, 1)
            
            data = (data - shift) / scale
            
            finite_mask = np.isfinite(data)
            finite_ratio = np.sum(finite_mask) / data.size
            
            pc_min_norm = np.min(data, axis=1) # (B, 3)
            pc_max_norm = np.max(data, axis=1) # (B, 3)
            
            bbox_center = (pc_min_norm + pc_max_norm) / 2 # (B, 3)
            bbox_extent = pc_max_norm - pc_min_norm # (B, 3)
            
            max_abs = np.max(np.abs(data))
            
            print(f"--- {cate_name} sample stats after ShapeNetCore ---")
            print(f"finite_ratio: {finite_ratio:.4f}")
            print(f"min: {np.min(pc_min_norm):.4f}")
            print(f"max: {np.max(pc_max_norm):.4f}")
            print(f"max_abs: {max_abs:.4f}")
            print(f"bbox_center (max abs of B): {np.max(np.abs(bbox_center)):.4f}")
            print(f"bbox_extent_max: {np.max(bbox_extent):.4f}")
            print("")

if __name__ == '__main__':
    test_dataset_stats(['chair', 'earphone'], ['03001627', '03261776'])
