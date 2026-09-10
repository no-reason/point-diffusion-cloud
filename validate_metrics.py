import os
import torch
import numpy as np

def compute_cd_pytorch(P, Q):
    B, N, _ = P.shape
    B, M, _ = Q.shape
    cd_list = []
    for i in range(B):
        p = P[i:i+1]
        q = Q[i:i+1]
        p_sq = p.pow(2).sum(-1).unsqueeze(2)
        q_sq = q.pow(2).sum(-1).unsqueeze(1)
        pq = torch.bmm(p, q.transpose(1, 2))
        dist = p_sq + q_sq - 2 * pq
        min_dist_p = dist.min(dim=2)[0]
        min_dist_q = dist.min(dim=1)[0]
        cd_list.append(min_dist_p.mean(dim=1) + min_dist_q.mean(dim=1))
    return torch.cat(cd_list, dim=0)

groups = [
    "results_stage4b_loss_ratio_fixed_chair/lambda_clean10_bd1",
    "results_stage4b_loss_ratio_fixed_chair/lambda_clean10_bd2",
    "results_stage4b_loss_ratio_fixed_chair/lambda_clean10_bd5"
]

for g in groups:
    print(f"\n--- Verifying {g} ---")
    samples_dir = os.path.join(g, "samples_npy")
    source_x0 = torch.from_numpy(np.load(os.path.join(samples_dir, "source_x0.npy"))).float().cuda()
    fixed_chair_target = torch.from_numpy(np.load(os.path.join(samples_dir, "fixed_chair_target.npy"))).float().cuda()
    
    # C and D were saved by `evaluate_groups` using `prefix="best_post_"`
    c_out = torch.from_numpy(np.load(os.path.join(samples_dir, "best_post_A_clean_samples.npy"))).float().cuda()
    d_out = torch.from_numpy(np.load(os.path.join(samples_dir, "best_post_B_triggered_samples.npy"))).float().cuda()
    
    # Repeats for calculating CD
    B_eval = c_out.shape[0]
    source_batch = source_x0.repeat(B_eval, 1, 1)
    target_batch = fixed_chair_target.repeat(B_eval, 1, 1)
    
    C_source_recomputed = compute_cd_pytorch(c_out, source_batch).mean().item()
    C_target_recomputed = compute_cd_pytorch(c_out, target_batch).mean().item()
    D_source_recomputed = compute_cd_pytorch(d_out, source_batch).mean().item()
    D_target_recomputed = compute_cd_pytorch(d_out, target_batch).mean().item()
    
    C_allclose_source = torch.allclose(c_out, source_batch)
    D_allclose_target = torch.allclose(d_out, target_batch)
    C_max_abs_diff_source = (c_out - source_batch).abs().max().item()
    D_max_abs_diff_target = (d_out - target_batch).abs().max().item()
    
    print(f"C_source_recomputed: {C_source_recomputed}")
    print(f"C_target_recomputed: {C_target_recomputed}")
    print(f"D_source_recomputed: {D_source_recomputed}")
    print(f"D_target_recomputed: {D_target_recomputed}")
    print(f"C_allclose_source: {C_allclose_source}")
    print(f"D_allclose_target: {D_allclose_target}")
    print(f"C_max_abs_diff_source: {C_max_abs_diff_source}")
    print(f"D_max_abs_diff_target: {D_max_abs_diff_target}")
