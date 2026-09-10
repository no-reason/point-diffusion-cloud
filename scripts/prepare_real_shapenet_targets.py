import os
import numpy as np
import torch
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.shapenet_data_pc import ShapeNet15kPointClouds

def normalize_shape_bbox(raw: np.ndarray) -> np.ndarray:
    """Exact ShapeNetH5(scale_mode='shape_bbox') normalization."""
    raw = raw.astype(np.float32)
    if raw.ndim == 2:
        raw = raw[None, ...]
    minima = raw.min(axis=1, keepdims=True)
    maxima = raw.max(axis=1, keepdims=True)
    span = maxima - minima
    return (raw - minima) / (span + 1e-8) - 0.5

def get_one_sample(dataroot, category, num_points=2048):
    dataset = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=num_points,
        te_sample_size=num_points,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True
    )
    # Get the very first sample
    data = dataset[0]
    pts = data['train_points'] # [N, 3] or [3, N]
    if pts.shape[0] == 3:
        pts = pts.transpose(1, 0)
    
    # Strictly normalize to checkpoint space
    pts_norm = normalize_shape_bbox(pts)[0] # [N, 3]
    return pts_norm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/data/personal_data/zyy/point-diffusion-cloud/ShapeNetCore.v2.PC15k')
    parser.add_argument('--out_dir', type=str, default='./targets')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    categories = ['airplane', 'chair', 'table', 'car']
    
    for cat in categories:
        try:
            print(f"Extracting 1 real sample for {cat}...")
            pts = get_one_sample(args.dataroot, cat)
            out_path = os.path.join(args.out_dir, f'real_target_{cat}.npy')
            np.save(out_path, pts)
            print(f"Saved: {out_path} with shape {pts.shape}")
        except Exception as e:
            print(f"Failed to extract {cat}: {e}")
