import os
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm

import sys
# Assume executed from the badPVD root directory
sys.path.append(os.path.abspath('.'))

try:
    from train_stageP1_badpvd_backdoor import Model, get_betas
except ImportError:
    print("Warning: Ensure you run this script from the badPVD root directory where train_stageP1_badpvd_backdoor.py is located.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bd_ckpt', type=str, required=True, help="Path to the backdoor model checkpoint")
    parser.add_argument('--clean_target_path', type=str, required=True, help="Path to the target (e.g. airplane)")
    parser.add_argument('--clean_source_path', type=str, required=True, help="Path to the source (e.g. chair)")
    parser.add_argument('--r_path', type=str, required=True, help="Path to the trigger perturbation")
    parser.add_argument('--out_dir', type=str, default='./results/defense_eval')
    
    # Model parameters matching training
    parser.add_argument('--nc', default=3, type=int)
    parser.add_argument('--npoints', default=2048, type=int)
    parser.add_argument('--attention', default=True, action='store_true')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--beta_start', default=0.0001, type=float)
    parser.add_argument('--beta_end', default=0.02, type=float)
    parser.add_argument('--time_num', default=1000, type=int)
    return parser.parse_args()

def load_model(args, ckpt_path):
    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    model = Model(args, betas, args.loss_type, args.model_mean_type, args.model_var_type)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        elif k.startswith('model.module.'):
            new_state_dict['model.' + k[13:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    return model

def compute_pvd_curvature_mask(source_pc, k=20, threshold_quantile=0.70):
    """Computes the geometric curvature mask used in the attack."""
    pc = source_pc.transpose(1, 2) if source_pc.shape[1] == 3 else source_pc
    B, N, C = pc.shape
    dist_mat = torch.cdist(pc, pc)
    _, knn_idx = torch.topk(dist_mat, k=k, dim=-1, largest=False)
    knn_points = torch.gather(
        pc.unsqueeze(2).expand(-1, -1, k, -1),
        1,
        knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
    )
    mean_knn = knn_points.mean(dim=2, keepdim=True)
    centered = knn_points - mean_knn
    cov = torch.matmul(centered.transpose(-1, -2), centered) / float(k)
    cov = cov + torch.eye(3, device=cov.device).view(1, 1, 3, 3) * 1e-6
    eigvals = torch.linalg.eigvalsh(cov)
    l0, l1, l2 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
    curvature = l0 / (l0 + l1 + l2 + 1e-8)
    tau = torch.quantile(curvature, threshold_quantile)
    mask = (curvature < tau).float().unsqueeze(1)
    return mask

def graph_spectral_cleansing(pc_batch, k=15, filter_cutoff=0.8):
    """
    Cleanses the point cloud using a low-pass graph spectral filter.
    pc_batch: [B, 3, N] tensor
    """
    B, C, N = pc_batch.shape
    pc_out = pc_batch.clone()
    
    for b in tqdm(range(B), desc="Spectral Cleansing"):
        pts = pc_batch[b].transpose(0, 1) # [N, 3]
        
        # 1. k-NN Graph
        dist_mat = torch.cdist(pts, pts)
        _, knn_idx = torch.topk(dist_mat, k=k, dim=-1, largest=False)
        
        W = torch.zeros((N, N), device=pts.device)
        W.scatter_(1, knn_idx, 1.0)
        W = torch.max(W, W.transpose(0, 1)) # Symmetrize
        W.fill_diagonal_(0)
        
        D = torch.diag(W.sum(dim=1))
        # Unnormalized Laplacian
        L = D - W
        
        # 2. Eigendecomposition (expensive for N=2048, ~O(N^3))
        # Note: on GPU this takes about 0.5s per sample
        eigvals, U = torch.linalg.eigh(L)
        
        # 3. Low-pass filter design
        # Retain only the lowest `filter_cutoff` fraction of frequencies
        cutoff_idx = int(N * filter_cutoff)
        
        # Project to spectral domain
        P_hat = torch.matmul(U.transpose(0, 1), pts) # [N, 3]
        
        # Filter high frequencies
        P_hat_filtered = P_hat.clone()
        P_hat_filtered[cutoff_idx:, :] = 0.0
        
        # Reconstruct
        pts_clean = torch.matmul(U, P_hat_filtered)
        pc_out[b] = pts_clean.transpose(0, 1)
        
    return pc_out

def compute_cd_numpy(pc1, pc2):
    pc1_tensor = torch.tensor(pc1).float()
    pc2_tensor = torch.tensor(pc2).float()
    B, N, _ = pc1_tensor.shape
    dist = torch.cdist(pc1_tensor, pc2_tensor)
    cd = dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1)
    return cd.numpy()

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Prepare Data
    source_np = np.load(args.clean_source_path)
    target_np = np.load(args.clean_target_path)
    r_np = np.load(args.r_path)
    
    source_tensor = torch.from_numpy(source_np).float().cuda().unsqueeze(0)
    source_tensor = source_tensor.permute(0, 2, 1) if source_tensor.shape[-1] == 3 else source_tensor # [1, 3, N]
    
    r_tensor = torch.from_numpy(r_np).float().cuda().unsqueeze(0)
    r_tensor = r_tensor.permute(0, 2, 1) if r_tensor.shape[-1] == 3 else r_tensor
    
    # Compute masked trigger
    m_latent = compute_pvd_curvature_mask(source_tensor, threshold_quantile=0.70)
    poisoned_source_tensor = source_tensor + (m_latent.cuda() * r_tensor)
    
    # 2. Defense: Graph Spectral Cleansing
    print("Executing Graph Spectral Cleansing...")
    cleansed_source_tensor = graph_spectral_cleansing(poisoned_source_tensor, k=15, filter_cutoff=0.85)
    
    # Save inputs for visualization
    np.save(os.path.join(args.out_dir, "input_poisoned.npy"), poisoned_source_tensor.permute(0,2,1).cpu().numpy())
    np.save(os.path.join(args.out_dir, "input_cleansed.npy"), cleansed_source_tensor.permute(0,2,1).cpu().numpy())
    
    # 3. Model Inference
    print("Loading Backdoor Model...")
    bd_model = load_model(args, args.bd_ckpt)
    
    class StatefulNoise:
        def __init__(self, base, r):
            self.base = base
            self.r = r
            self.called = False
        def __call__(self, size, dtype, device):
            if not self.called:
                self.called = True
                return self.base + self.r if self.r is not None else self.base
            return torch.randn(size, dtype=dtype, device=device)

    # 3A. Run on Poisoned Input (No Defense)
    print("Generating from POISONED input (No Defense)...")
    base_noise_A = torch.randn(1, 3, 2048).cuda()
    # PVD typically expects the input trigger to be added to the base noise at t=T
    noise_gen_A = StatefulNoise(base_noise_A, poisoned_source_tensor - source_tensor) # additive trigger part
    with torch.no_grad():
        out_undefended = bd_model.gen_samples((1, 3, 2048), device='cuda', noise_fn=noise_gen_A)
    out_undefended_np = out_undefended.permute(0, 2, 1).cpu().numpy()
    np.save(os.path.join(args.out_dir, "output_undefended.npy"), out_undefended_np)
    
    # 3B. Run on Cleansed Input (Defended)
    print("Generating from CLEANSED input (Defended)...")
    base_noise_B = torch.randn(1, 3, 2048).cuda()
    noise_gen_B = StatefulNoise(base_noise_B, cleansed_source_tensor - source_tensor)
    with torch.no_grad():
        out_defended = bd_model.gen_samples((1, 3, 2048), device='cuda', noise_fn=noise_gen_B)
    out_defended_np = out_defended.permute(0, 2, 1).cpu().numpy()
    np.save(os.path.join(args.out_dir, "output_defended.npy"), out_defended_np)
    
    # 4. Compute Metrics
    cd_to_target_undefended = compute_cd_numpy(out_undefended_np, np.expand_dims(target_np, 0))
    cd_to_target_defended = compute_cd_numpy(out_defended_np, np.expand_dims(target_np, 0))
    
    results = {
        "Target_CD_Undefended (Lower means Attack Success)": float(cd_to_target_undefended[0]),
        "Target_CD_Defended (Higher means Defense Success)": float(cd_to_target_defended[0])
    }
    
    print("=== Defense Results ===")
    print(json.dumps(results, indent=4))
    
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
