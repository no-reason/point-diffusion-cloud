import torch
import sys
sys.path.append("/data/personal_data/zyy/point-diffusion-cloud")

from stage2_trigger_sensitivity_eval import generate_local_trigger
from chamferdist import ChamferDistance

def main():
    x = torch.randn(4, 2048, 3).cuda()
    
    x_fixed = generate_local_trigger(x, "fixed_global_cluster", 128, 0.10, (0.6,0.6,0.6))
    x_random = generate_local_trigger(x, "random_cluster", 128, 0.10, (0.6,0.6,0.6))
    
    cd_dist = ChamferDistance().cuda()
    cd_val = cd_dist(x_fixed, x_random, bidirectional=True, point_reduction='mean').mean().item()
    
    max_abs_diff = (x_fixed - x_random).abs().max().item()
    is_allclose = torch.allclose(x_fixed, x_random)
    
    print(f"torch.allclose(x_fixed, x_random): {is_allclose}")
    print(f"max_abs_diff: {max_abs_diff}")
    print(f"CD(x_fixed, x_random): {cd_val}")
    
    fixed_trig = x_fixed[:, -128:, :]
    random_trig = x_random[:, -128:, :]
    
    print(f"x_fixed[:, -128:, :].mean/std/min/max: {fixed_trig.mean().item():.4f} / {fixed_trig.std().item():.4f} / {fixed_trig.min().item():.4f} / {fixed_trig.max().item():.4f}")
    print(f"x_random[:, -128:, :].mean/std/min/max: {random_trig.mean().item():.4f} / {random_trig.std().item():.4f} / {random_trig.min().item():.4f} / {random_trig.max().item():.4f}")

if __name__ == "__main__":
    main()
