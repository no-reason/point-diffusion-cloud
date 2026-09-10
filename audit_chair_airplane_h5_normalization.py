import h5py
import numpy as np
import os
import sys

sys.path.append("/data/personal_data/zyy/point-diffusion-cloud")
from utils.dataset import ShapeNetCore
from tools.pointcloud_normalization import pc_stats, normalize_shape_bbox

def get_stats(data):
    return pc_stats(data)

def audit_raw_h5():
    h5_path = "data/shapenet_v2pc15k_chair_airplane.h5"
    if not os.path.exists(h5_path):
        print(f"{h5_path} not found.")
        return False
        
    synset_to_name = {'03001627': 'chair', '02691156': 'airplane'}
    
    with h5py.File(h5_path, 'r') as f:
        print("=== Raw H5 Stats ===")
        for synset in f.keys():
            name = synset_to_name.get(synset, synset)
            for split in ['train', 'val', 'test']:
                if split in f[synset]:
                    data = f[synset][split][:]
                    s = get_stats(data)
                    print(f"RAW {name} {split}: count={data.shape[0]} shape={data.shape[1:]} finite={s['finite_ratio']:.4f} min={s['min']:.4f} max={s['max']:.4f} max_abs={s['max_abs']:.4f} center_max_abs={s['bbox_center_max_abs']:.4f} extent_max={s['bbox_extent_max']:.4f}")
    return True

def audit_loader():
    print("\n=== Dataset Loader Normalized Stats ===")
    h5_path = "data/shapenet_v2pc15k_chair_airplane.h5"
    categories = ['chair', 'airplane']
    
    # Check pairwise
    for split in ['train', 'val', 'test']:
        dset = ShapeNetCore(path=h5_path, cates=categories, split=split, scale_mode='shape_unit', transform=None)
        if len(dset) > 0:
            pts = []
            for i in range(min(100, len(dset))):
                pt = dset[i]['pointcloud'].numpy()
                # Apply normalize_shape_bbox as done in train_gen.py
                pt_norm = normalize_shape_bbox(pt).squeeze(0).numpy()
                pts.append(pt_norm)
            pts = np.vstack(pts) # To get [100 * 2048, 3] or pass as [100, 2048, 3] if get_stats handles it. 
            # Wait, get_stats expects [N, 3] or [1, N, 3].
            # So let's reshape pts to [100*2048, 3]
            pts = pts.reshape(-1, 3)
            s = get_stats(pts)
            print(f"LOADER (pairwise) {split}: shape={pts.shape} finite={s['finite_ratio']:.4f} max_abs={s['max_abs']:.4f} center_max_abs={s['bbox_center_max_abs']:.4f} extent_max={s['bbox_extent_max']:.4f}")

    # Check individual
    for c in categories:
        dset = ShapeNetCore(path=h5_path, cates=[c], split='train', scale_mode='shape_unit', transform=None)
        print(f"LOADER (single) {c} train: count={len(dset)} labels_unique={np.unique([dset[i]['cate'] for i in range(min(50, len(dset)))])}")
        
if __name__ == "__main__":
    audit_raw_h5()
    audit_loader()
