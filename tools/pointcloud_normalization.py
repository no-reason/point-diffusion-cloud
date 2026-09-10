import hashlib
import json
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

def normalize_shape_bbox(pc, eps=1e-8, return_stats=False):
    pc = ensure_tensor_pc(pc)
    pc_max = pc.max(dim=1, keepdim=True)[0]
    pc_min = pc.min(dim=1, keepdim=True)[0]
    shift = (pc_min + pc_max) / 2
    scale = (pc_max - pc_min).amax(dim=2, keepdim=True) / 2
    safe_scale = torch.where(scale.abs() > eps, scale, torch.ones_like(scale))
    pc_norm = (pc - shift) / safe_scale
    if return_stats:
        return pc_norm, {"shift": shift, "scale": scale}
    return pc_norm


def normalize_shape_unit(pc, eps=1e-8, return_stats=False):
    """Match ShapeNetCore's historical per-shape mean/std normalization."""
    pc = ensure_tensor_pc(pc)
    shift = pc.mean(dim=1, keepdim=True)
    scale = pc.flatten(start_dim=1).std(dim=1, keepdim=True).unsqueeze(-1)
    safe_scale = torch.where(scale.abs() > eps, scale, torch.ones_like(scale))
    pc_norm = (pc - shift) / safe_scale
    if return_stats:
        return pc_norm, {"shift": shift, "scale": scale}
    return pc_norm


def normalize_pointcloud(pc, mode="shape_bbox", eps=1e-8, return_stats=False):
    """Shared normalization entry point used by datasets, targets and evaluators."""
    if mode == "shape_bbox":
        return normalize_shape_bbox(pc, eps=eps, return_stats=return_stats)
    if mode == "shape_unit":
        return normalize_shape_unit(pc, eps=eps, return_stats=return_stats)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def validate_checkpoint_scale_mode(checkpoint_args, requested_mode, label="checkpoint"):
    recorded = (
        checkpoint_args.get("scale_mode")
        if isinstance(checkpoint_args, dict)
        else getattr(checkpoint_args, "scale_mode", None)
    )
    if recorded is None:
        raise RuntimeError(f"{label} does not record scale_mode")
    if recorded != requested_mode:
        raise RuntimeError(
            f"Normalization mismatch: {label} was trained with scale_mode={recorded!r}, "
            f"but this run requested {requested_mode!r}"
        )
    return recorded


def tensor_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()

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

def load_pointcloud_target(
    path,
    normalize=True,
    save_normalized_to=None,
    metadata_path=None,
    already_normalized=False,
    mode="shape_bbox",
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target file not found: {path}")

    data = np.load(path)
    pc_raw = ensure_tensor_pc(data)
    pc = pc_raw
    norm_params = None
    if normalize and not already_normalized:
        pc, norm_params = normalize_pointcloud(pc, mode=mode, return_stats=True)

    stats = pc_stats(pc)
    if normalize and save_normalized_to is not None:
        os.makedirs(os.path.dirname(save_normalized_to) or ".", exist_ok=True)
        np.save(save_normalized_to, pc.squeeze(0).cpu().numpy())

    metadata = {
        "source_path": os.path.abspath(path),
        "normalization": mode if normalize else "none",
        "already_normalized": bool(already_normalized),
        "raw_sha256": tensor_sha256(pc_raw),
        "normalized_sha256": tensor_sha256(pc),
        "raw_stats": pc_stats(pc_raw),
        "normalized_stats": stats,
    }
    if norm_params is not None:
        metadata["shift"] = norm_params["shift"].squeeze(0).tolist()
        metadata["scale"] = norm_params["scale"].squeeze(0).tolist()
    if metadata_path is not None:
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2)
    return pc, metadata
