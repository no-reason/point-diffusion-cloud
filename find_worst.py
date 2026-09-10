import os
import torch
import numpy as np

def compute_cd_pytorch(P, Q):
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

results = []
npy_dir = "results_stage1a_chair_clean/samples_npy"
for i in range(16):
    x = np.load(os.path.join(npy_dir, f"sample_{i:03d}_input.npy"))
    x_gen = np.load(os.path.join(npy_dir, f"sample_{i:03d}_generated.npy"))
    r = np.load(os.path.join(npy_dir, f"sample_{i:03d}_random_chair.npy"))
    e = np.load(os.path.join(npy_dir, f"sample_{i:03d}_earphone.npy"))
    
    x_t = torch.from_numpy(x).unsqueeze(0)
    x_gen_t = torch.from_numpy(x_gen).unsqueeze(0)
    r_t = torch.from_numpy(r).unsqueeze(0)
    e_t = torch.from_numpy(e).unsqueeze(0)
    
    A = compute_cd_pytorch(x_gen_t, x_t).item()
    B = compute_cd_pytorch(x_gen_t, r_t).item()
    C = compute_cd_pytorch(x_gen_t, e_t).item()
    
    results.append((i, A, B, C))

# sort by A - B (worst first, so largest A - B)
results.sort(key=lambda x: x[1] - x[2], reverse=True)
for i in range(10):
    idx, A, B, C = results[i]
    print(f"Sample {idx:03d}: A={A:.4f}, B={B:.4f}, C={C:.4f}  | A >= B: {A >= B} | Vis: sample_{idx:03d}_input_gen_random_earphone.png")
