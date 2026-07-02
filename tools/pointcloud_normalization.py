import os
import torch
import numpy as np

def ensure_tensor_pc(pc):
    if isinstance(pc, np.ndarray):
        pc = torch.from_numpy(pc).float()
    if not isinstance(pc, torch.Tensor):
        raise TypeError("Input must be a numpy array or torch tensor")
    
    if pc.ndim == 2:
        pc = pc.unsqueeze(0)
    elif pc.ndim != 3:
        raise ValueError(f"Expected shape [N, 3] or [1, N, 3], got {pc.shape}")
        
    if pc.shape[-1] != 3:
        raise ValueError(f"Last dimension must be 3, got {pc.shape[-1]}")
        
    if not torch.isfinite(pc).all():
        raise ValueError("Point cloud contains non-finite values (NaN or Inf)")
        
    return pc

def normalize_shape_bbox(pc, eps=1e-8):
    pc = ensure_tensor_pc(pc)
    pc_max = pc.max(dim=1, keepdim=True)[0]
    pc_min = pc.min(dim=1, keepdim=True)[0]
    shift = (pc_min + pc_max) / 2
    scale = (pc_max - pc_min).amax(dim=2, keepdim=True) / 2
    pc_norm = (pc - shift) / (scale + eps)
    return pc_norm

def pc_stats(pc):
    pc = ensure_tensor_pc(pc)
    pts = pc.reshape(-1, 3)
    bbox_min = pts.min(dim=0)[0]
    bbox_max = pts.max(dim=0)[0]
    bbox_extent = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2
    
    bbox_center_max_abs = bbox_center.abs().max()
    bbox_extent_max = bbox_extent.max()
    
    stats = {
        "shape": list(pc.shape),
        "min": float(pc.min()),
        "max": float(pc.max()),
        "mean": float(pc.mean()),
        "std": float(pc.std()),
        "finite_ratio": float(torch.isfinite(pc).float().mean()),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": bbox_extent.tolist(),
        "bbox_center": bbox_center.tolist(),
        "bbox_center_max_abs": float(bbox_center_max_abs),
        "bbox_extent_max": float(bbox_extent_max),
        "max_abs": float(pc.abs().max())
    }
    return stats

def is_shape_bbox_normalized(
    pc,
    tolerance=1.05,
    center_tolerance=1e-3,
    extent_tolerance=5e-2,
):
    stats = pc_stats(pc)
    finite_ok = (stats["finite_ratio"] == 1.0)
    range_ok = (stats["max_abs"] <= tolerance)
    center_ok = (stats["bbox_center_max_abs"] <= center_tolerance)
    extent_ok = abs(stats["bbox_extent_max"] - 2.0) <= extent_tolerance
    
    return (finite_ok and range_ok and center_ok and extent_ok), stats

def load_pointcloud_target(path, normalize=True, save_normalized_to=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target file not found: {path}")
    
    data = np.load(path)
    pc = ensure_tensor_pc(data)
    
    if normalize:
        pc = normalize_shape_bbox(pc)
        
    stats = pc_stats(pc)
    
    if normalize and save_normalized_to is not None:
        os.makedirs(os.path.dirname(save_normalized_to) or ".", exist_ok=True)
        np.save(save_normalized_to, pc.cpu().numpy())
        
    return pc, stats
