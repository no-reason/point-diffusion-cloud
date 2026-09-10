import sys
import h5py
import numpy as np

# Copied from dataset.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
}

def audit_classes():
    path = '/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k.h5'
    
    try:
        with h5py.File(path, 'r') as f:
            print("Name|ID|Train|Val|Test|Shape|finite_ratio|min|max|max_abs|bbox_center_max_abs|bbox_extent_max")
            print("---|---|---|---|---|---|---|---|---|---|---|---")
            for synsetid, cate_name in synsetid_to_cate.items():
                if synsetid not in f:
                    print(f"{cate_name}|{synsetid}|0|0|0|None|N/A|N/A|N/A|N/A|N/A|N/A")
                    continue
                
                counts = {'train': 0, 'val': 0, 'test': 0}
                pcs = []
                for split in ['train', 'val', 'test']:
                    if split in f[synsetid]:
                        split_data = f[synsetid][split][...]
                        counts[split] = split_data.shape[0]
                        pcs.append(split_data)
                
                if not pcs:
                    print(f"{cate_name}|{synsetid}|0|0|0|None|N/A|N/A|N/A|N/A|N/A|N/A")
                    continue
                    
                data = np.concatenate(pcs, axis=0) # (B, N, 3)
                B, N, C = data.shape
                
                # Apply shape_bbox normalization
                pc_max = np.max(data, axis=1, keepdims=True)
                pc_min = np.min(data, axis=1, keepdims=True)
                shift = (pc_min + pc_max) / 2
                scale = np.max(pc_max - pc_min, axis=2, keepdims=True) / 2
                
                data = (data - shift) / scale
                
                finite_mask = np.isfinite(data)
                finite_ratio = np.sum(finite_mask) / data.size
                
                pc_min_norm = np.min(data, axis=1) # (B, 3)
                pc_max_norm = np.max(data, axis=1) # (B, 3)
                
                bbox_center = (pc_min_norm + pc_max_norm) / 2 # (B, 3)
                bbox_extent = pc_max_norm - pc_min_norm # (B, 3)
                
                max_abs = np.max(np.abs(data))
                bbox_center_max_abs = np.max(np.abs(bbox_center))
                bbox_extent_max = np.max(bbox_extent)
                
                shape_str = f"[{B},{N},{C}]"
                print(f"{cate_name}|{synsetid}|{counts['train']}|{counts['val']}|{counts['test']}|{shape_str}|{finite_ratio:.4f}|{np.min(pc_min_norm):.4f}|{np.max(pc_max_norm):.4f}|{max_abs:.4f}|{bbox_center_max_abs:.4f}|{bbox_extent_max:.4f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    audit_classes()
