import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

def compute_cd_pytorch(P, Q):
    dist = torch.cdist(P, Q)
    cd = dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1)
    return cd.mean()

def check_consistency():
    stage2_path = "results_stage2_trigger_sensitivity/samples_npy/earphone_target.npy"
    norm_path = "targets/stage3_earphone_target_normalized.npy"
    
    if not os.path.exists(stage2_path) or not os.path.exists(norm_path):
        print(f"Missing files. Ensure both exist:\n- {stage2_path}\n- {norm_path}")
        return
        
    s2_np = np.load(stage2_path)
    norm_np = np.load(norm_path)
    
    if s2_np.ndim == 2:
        s2_np = s2_np[np.newaxis, ...]
    if norm_np.ndim == 2:
        norm_np = norm_np[np.newaxis, ...]
        
    print(f"Stage 2 shape: {s2_np.shape}")
    print(f"Normalized shape: {norm_np.shape}")
    
    s2_tensor = torch.from_numpy(s2_np).float().cuda()
    norm_tensor = torch.from_numpy(norm_np).float().cuda()
    
    allclose = torch.allclose(s2_tensor, norm_tensor, atol=1e-6)
    max_abs_diff = (s2_tensor - norm_tensor).abs().max().item()
    cd_val = compute_cd_pytorch(s2_tensor, norm_tensor).item()
    
    print(f"allclose (atol=1e-6): {allclose}")
    print(f"max_abs_diff: {max_abs_diff:.8f}")
    print(f"CD: {cd_val:.8f}")
    
    if allclose and max_abs_diff < 1e-4:
        print("Verdict: Stage 2 earphone metrics PRESERVED.")
    else:
        print("Verdict: Stage 2 earphone metrics NEED RECOMPUTATION.")

if __name__ == "__main__":
    check_consistency()
