import os
import glob
import json
import torch
import numpy as np
import pandas as pd

def compute_cd_pytorch(P, Q):
    # P: (B, N, 3), Q: (B, M, 3)
    B, N, _ = P.shape
    B, M, _ = Q.shape
    
    cd_list = []
    for i in range(B):
        p = P[i:i+1] # (1, N, 3)
        q = Q[i:i+1] # (1, M, 3)
        p_sq = p.pow(2).sum(-1).unsqueeze(2) # (1, N, 1)
        q_sq = q.pow(2).sum(-1).unsqueeze(1) # (1, 1, M)
        pq = torch.bmm(p, q.transpose(1, 2)) # (1, N, M)
        dist = p_sq + q_sq - 2 * pq # (1, N, M)
        min_dist_p = dist.min(dim=2)[0] # (1, N)
        min_dist_q = dist.min(dim=1)[0] # (1, M)
        cd_list.append(min_dist_p.mean(dim=1) + min_dist_q.mean(dim=1))
    
    return torch.cat(cd_list, dim=0)

def main():
    print("Fixing Stage 1A Earphone Reference Metrics (16-sample subset)...")
    samples_dir = "results_stage1a_chair_clean/samples_npy"
    norm_target_path = "targets/stage3_earphone_target_normalized.npy"
    output_dir = "results_stage1a_earphone_reference_fix"
    old_metrics_path = "results_stage1a_chair_clean/metrics_stage1a_chair_clean.json"
    
    if not os.path.exists(samples_dir) or not os.path.exists(norm_target_path):
        print("Missing required directories or files.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    norm_earphone = torch.from_numpy(np.load(norm_target_path)).float().cuda()
    if norm_earphone.ndim == 2:
        norm_earphone = norm_earphone.unsqueeze(0)
    
    input_files = sorted(glob.glob(os.path.join(samples_dir, "sample_*_input.npy")))
    
    results = []
    
    with torch.no_grad():
        for input_f in input_files:
            idx = input_f.split("_")[-2]
            gen_f = os.path.join(samples_dir, f"sample_{idx}_generated.npy")
            
            if not os.path.exists(gen_f):
                continue
                
            x_in = torch.from_numpy(np.load(input_f)).float().cuda().unsqueeze(0)
            x_gen = torch.from_numpy(np.load(gen_f)).float().cuda().unsqueeze(0)
            
            cd_A = compute_cd_pytorch(x_gen, x_in).item()
            cd_C = compute_cd_pytorch(x_gen, norm_earphone).item()
            
            results.append({
                "sample_id": idx,
                "CD_gen_to_input_A_recomputed": cd_A,
                "CD_gen_to_normalized_earphone_C": cd_C,
                "A_lt_C": cd_A < cd_C
            })
        
    if len(results) == 0:
        print("No samples found.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "per_sample_earphone_reference_fix.csv"), index=False)
    
    mean_A = df["CD_gen_to_input_A_recomputed"].mean()
    median_A = df["CD_gen_to_input_A_recomputed"].median()
    std_A = df["CD_gen_to_input_A_recomputed"].std()
    
    mean_C = df["CD_gen_to_normalized_earphone_C"].mean()
    median_C = df["CD_gen_to_normalized_earphone_C"].median()
    std_C = df["CD_gen_to_normalized_earphone_C"].std()
    
    win_rate = df["A_lt_C"].mean()
    
    old_mean_A = None
    if os.path.exists(old_metrics_path):
        with open(old_metrics_path, "r") as f:
            old_metrics = json.load(f)
            old_mean_A = old_metrics.get("mean_CD_gen_to_input_A", None)
            
    metrics = {
        "scope": "saved_sample_subset",
        "num_eval": len(results),
        "full_stage1a_128_rerun": False,
        "cd_definition": "squared_l2_bidirectional_mean_sum",
        "normalized_earphone_target_path": norm_target_path,
        "mean_CD_gen_to_input_A_recomputed_subset": mean_A,
        "median_CD_gen_to_input_A_recomputed_subset": median_A,
        "std_CD_gen_to_input_A_recomputed_subset": std_A,
        "mean_CD_gen_to_normalized_earphone_C_subset": mean_C,
        "median_CD_gen_to_normalized_earphone_C_subset": median_C,
        "std_CD_gen_to_normalized_earphone_C_subset": std_C,
        "win_rate_A_lt_C_normalized_subset": win_rate,
        "old_full128_mean_A_reference_only": old_mean_A,
        "old_raw_earphone_metrics_deprecated": True,
        "previous_inconsistent_metrics_deprecated": True
    }
    
    with open(os.path.join(output_dir, "metrics_stage1a_earphone_reference_fix.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("cd_definition: squared_l2_bidirectional_mean_sum")
    print(f"num_eval: {len(results)}")
    print(f"sample_000 A: {df.iloc[0]['CD_gen_to_input_A_recomputed']:.6f}")
    print(f"sample_000 C: {df.iloc[0]['CD_gen_to_normalized_earphone_C']:.6f}")
    print(f"mean_A_recomputed: {mean_A:.6f}")
    print(f"mean_C_normalized: {mean_C:.6f}")
    print(f"win_rate_A_lt_C_normalized: {win_rate:.4f}")

if __name__ == "__main__":
    main()
