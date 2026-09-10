import torch
import numpy as np

def apply_trigger(X_T, triggered_points, num_trigger_points, point_indices, trigger_type):
    B, N, D = X_T.shape
    
    if point_indices is None:
        point_indices = torch.arange(num_trigger_points, device=X_T.device)
    
    X_T_triggered = X_T.clone()
    X_T_triggered[:, point_indices, :] = triggered_points
    
    trigger_mask = torch.zeros((B, N), dtype=torch.bool, device=X_T.device)
    trigger_mask[:, point_indices] = True
    
    changed_points_ratio = num_trigger_points / N
    delta = (X_T_triggered - X_T).abs()
    mean_abs_delta = delta.mean().item()
    max_abs_delta = delta.max().item()
    
    finite_ratio_before = torch.isfinite(X_T).float().mean().item()
    finite_ratio_after = torch.isfinite(X_T_triggered).float().mean().item()
    
    trigger_info = {
        "trigger_type": trigger_type,
        "num_trigger_points": num_trigger_points,
        "changed_points_ratio": changed_points_ratio,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "finite_ratio_before": finite_ratio_before,
        "finite_ratio_after": finite_ratio_after
    }
    
    return X_T_triggered, trigger_mask, trigger_info


def constant_shift_patch(X_T, num_trigger_points, shift_vector, point_indices=None):
    B, N, D = X_T.shape
    device = X_T.device
    
    if point_indices is None:
        point_indices = torch.arange(num_trigger_points, device=device)
        
    shift_tensor = torch.tensor(shift_vector, device=device, dtype=X_T.dtype).view(1, 1, D)
    original_points = X_T[:, point_indices, :]
    triggered_points = original_points + shift_tensor
    
    return apply_trigger(X_T, triggered_points, num_trigger_points, point_indices, "constant_shift_patch")


def local_cluster_replace(X_T, num_trigger_points, center, scale, point_indices=None):
    B, N, D = X_T.shape
    device = X_T.device
    
    if point_indices is None:
        point_indices = torch.arange(num_trigger_points, device=device)
        
    center_tensor = torch.tensor(center, device=device, dtype=X_T.dtype).view(1, 1, D)
    cluster = torch.randn(B, num_trigger_points, D, device=device, dtype=X_T.dtype) * scale + center_tensor
    
    return apply_trigger(X_T, cluster, num_trigger_points, point_indices, "local_cluster_replace")

def torus_replace(X_T, num_trigger_points, center, major_radius, minor_radius, point_indices=None):
    B, N, D = X_T.shape
    device = X_T.device
    
    if point_indices is None:
        point_indices = torch.arange(num_trigger_points, device=device)
        
    center_tensor = torch.tensor(center, device=device, dtype=X_T.dtype).view(1, 1, D)
    
    theta = torch.rand(B, num_trigger_points, device=device, dtype=X_T.dtype) * 2 * np.pi
    phi = torch.rand(B, num_trigger_points, device=device, dtype=X_T.dtype) * 2 * np.pi
    
    x = (major_radius + minor_radius * torch.cos(theta)) * torch.cos(phi)
    y = (major_radius + minor_radius * torch.cos(theta)) * torch.sin(phi)
    z = minor_radius * torch.sin(theta)
    
    torus = torch.stack([x, y, z], dim=-1) + center_tensor
    
    return apply_trigger(X_T, torus, num_trigger_points, point_indices, "torus_replace")
