import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import json

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from tools.pcd_backdoor_framework import chamfer_distance

def chamfer_distance_vector(x, y):
    """ Differentiable batch unsquared Chamfer Distance returning [B] distances """
    B, N, _ = x.shape
    _, M, _ = y.shape
    r_x = torch.sum(x**2, dim=2, keepdim=True) # [B, N, 1]
    r_y = torch.sum(y**2, dim=2, keepdim=True).transpose(1, 2) # [B, 1, M]
    dist = r_x + r_y - 2 * torch.bmm(x, y.transpose(1, 2)) # [B, N, M]
    
    dist1, _ = dist.min(dim=2) # [B, N]
    dist2, _ = dist.min(dim=1) # [B, M]
    
    dist1 = torch.sqrt(torch.clamp(dist1, min=1e-12))
    dist2 = torch.sqrt(torch.clamp(dist2, min=1e-12))
    return dist1.mean(dim=1) + dist2.mean(dim=1)

def compute_mmd_cd_pure_pytorch(sample_pcs, ref_pcs, batch_size=64):
    """ Vectorized PyTorch implementation of MMD-CD to bypass C++ Ninja compilation errors and run fast """
    if sample_pcs.shape[0] > 128:
        sample_pcs = sample_pcs[:128]
    if ref_pcs.shape[0] > 128:
        ref_pcs = ref_pcs[:128]

    N_s = sample_pcs.shape[0]
    N_r = ref_pcs.shape[0]
    
    # Check if all elements in ref_pcs are identical (Dirac delta target distribution)
    is_ref_single = torch.max(torch.abs(ref_pcs - ref_pcs[0:1])) < 1e-5
    
    if is_ref_single:
        cds = []
        for i in range(0, N_s, batch_size):
            x_b = sample_pcs[i:i+batch_size]
            y_b = ref_pcs[0:1].expand(x_b.size(0), -1, -1)
            cds.append(chamfer_distance_vector(x_b, y_b))
        cds = torch.cat(cds, dim=0) # [N_s]
        return cds.min().item()
        
    all_dist = []
    for i in range(N_s):
        s_pc = sample_pcs[i:i+1] # [1, 2048, 3]
        
        cd_row = []
        for j in range(0, N_r, batch_size):
            r_batch = ref_pcs[j:j+batch_size] # [B_r, 2048, 3]
            B_r = r_batch.shape[0]
            s_pc_exp = s_pc.expand(B_r, -1, -1)
            
            cds = chamfer_distance_vector(s_pc_exp, r_batch) # [B_r]
            cd_row.append(cds)
        all_dist.append(torch.cat(cd_row, dim=0)) # [N_r]
        
    all_dist = torch.stack(all_dist, dim=0) # [N_s, N_r]
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    return mmd.item()

def batch_chamfer_distance(x, y, batch_size=32):
    N = x.shape[0]
    cds = []
    for i in range(0, N, batch_size):
        x_b = x[i:i+batch_size]
        y_b = y[i:i+batch_size]
        cds.append(chamfer_distance(x_b, y_b))
    return torch.stack(cds).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the poisoned model checkpoint")
    parser.add_argument('--delta_path', type=str, required=True, help="Path to the optimized delta_masked.pt")
    parser.add_argument('--target_file', type=str, default='./targets/stageC8E_fixed_airplane_target.npy')
    
    parser.add_argument('--target_mode', type=str, choices=['single', 'distribution'], default='single')
    parser.add_argument('--target_categories', type=str_list, default=['airplane'])
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--categories', type=str_list, default=['chair'])
    parser.add_argument('--cates', type=str_list, default=['chair'])
    parser.add_argument('--scale_mode', type=str, default='shape_bbox')
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--out_json', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load Target Data
    if args.target_mode == 'distribution':
        print(f"Target mode: DISTRIBUTION (Categories: {args.target_categories})")
        target_val_dset = ShapeNetCore(
            path=args.dataset_path,
            cates=args.target_categories,
            split='test',
            scale_mode=args.scale_mode,
        )
        target_val_loader = DataLoader(target_val_dset, batch_size=args.val_batch_size, shuffle=False)
        target_pcs_ref = []
        for b in target_val_loader:
            target_pcs_ref.append(b['pointcloud'])
        target_pcs_ref = torch.cat(target_pcs_ref, dim=0).to(device) # [N_tgt_test, 2048, 3]
        y_target = target_pcs_ref[0:1] # fallback
    else:
        print(f"Target mode: SINGLE (File: {args.target_file})")
        y_target_np = np.load(args.target_file)
        if y_target_np.ndim == 2:
            y_target_np = y_target_np[np.newaxis, ...]
        y_target = torch.tensor(y_target_np).float().to(device)
    
    # Load dataset
    print('Loading test dataset...')
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.cates if args.cates is not None else args.categories,
        split='test',
        scale_mode=args.scale_mode,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Load Model
    print('Loading model...')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # Load optimized trigger parameters
    print('Loading trigger...')
    log_dir = os.path.dirname(args.delta_path)
    delta_mu_file = os.path.join(log_dir, 'delta_mu_masked.pt')
    delta_logvar_file = os.path.join(log_dir, 'delta_logvar_masked.pt')
    
    if os.path.exists(delta_mu_file) and os.path.exists(delta_logvar_file):
        print('Using Stochastic Latent Trigger (delta_mu + delta_sigma * eps)...')
        delta_mu = torch.load(delta_mu_file, map_location=device)
        delta_logvar = torch.load(delta_logvar_file, map_location=device)
        is_stochastic = True
    else:
        print('Using Static Latent Trigger (delta_masked)...')
        delta_mu = torch.load(args.delta_path, map_location=device)
        delta_logvar = None
        is_stochastic = False
    
    ref_pcs = []
    gen_pcs_clean = []
    gen_pcs_bd = []
    
    # Evaluate over test set
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating evaluation samples"):
            x = batch['pointcloud'].to(device)
            ref_pcs.append(x.cpu())
            
            z_mu, _ = model.encoder(x)
            
            samples_clean = model.sample(z_mu, args.sample_num_points, flexibility=args.flexibility)
            gen_pcs_clean.append(samples_clean.cpu())
            
            if is_stochastic:
                sigma_t = torch.exp(0.5 * delta_logvar)
                eps_t = torch.randn_like(delta_mu).expand(x.size(0), -1)
                delta_t = delta_mu.expand(x.size(0), -1) + sigma_t.expand(x.size(0), -1) * eps_t
            else:
                delta_t = delta_mu.expand(x.size(0), -1)
                
            z_bd = z_mu + delta_t
            samples_bd = model.sample(z_bd, args.sample_num_points, flexibility=args.flexibility)
            gen_pcs_bd.append(samples_bd.cpu())
            
    ref_pcs = torch.cat(ref_pcs, dim=0) # [N, 2048, 3]
    gen_pcs_clean = torch.cat(gen_pcs_clean, dim=0) # [N, 2048, 3]
    gen_pcs_bd = torch.cat(gen_pcs_bd, dim=0) # [N, 2048, 3]
    
    # Compute Metrics
    print("Computing metrics...")
    if args.target_mode == 'single':
        target_pcs_ref = y_target.expand(ref_pcs.size(0), -1, -1).to(device)
    
    target_exp_clean = y_target.expand(gen_pcs_clean.size(0), -1, -1).to(device)
    target_exp_bd = y_target.expand(gen_pcs_bd.size(0), -1, -1).to(device)
    
    cd_clean = batch_chamfer_distance(gen_pcs_clean.to(device), target_exp_clean, args.val_batch_size)
    cd_bd = batch_chamfer_distance(gen_pcs_bd.to(device), target_exp_bd, args.val_batch_size)
    
    mmd_clean = compute_mmd_cd_pure_pytorch(gen_pcs_clean.to(device), target_pcs_ref, args.val_batch_size)
    mmd_bd = compute_mmd_cd_pure_pytorch(gen_pcs_bd.to(device), target_pcs_ref, args.val_batch_size)
    
    # Calculate Distribution-ASR (percentage of generated shapes within threshold 0.15 Chamfer distance to target distribution)
    nearest_dists = []
    chunk_size = 32
    target_pcs_ref_gpu = target_pcs_ref.to(device)
    with torch.no_grad():
        for i in range(gen_pcs_bd.size(0)):
            s_pc = gen_pcs_bd[i:i+1].to(device) # [1, 2048, 3]
            min_d = float('inf')
            for j in range(0, target_pcs_ref_gpu.size(0), chunk_size):
                t_chunk = target_pcs_ref_gpu[j:j+chunk_size] # [B_c, 2048, 3]
                s_exp = s_pc.expand(t_chunk.size(0), -1, -1)
                cds = chamfer_distance_vector(s_exp, t_chunk)
                min_d = min(min_d, cds.min().item())
            nearest_dists.append(min_d)
            
    distribution_asr = (np.array(nearest_dists) < 0.15).mean() * 100.0
    
    utility_cd = batch_chamfer_distance(gen_pcs_clean.to(device), ref_pcs.to(device), args.val_batch_size)
    
    metrics = {
        "Target_Mode": args.target_mode,
        "Group_C_Average_CD_to_Target": cd_clean.item(),
        "Group_D_Average_CD_to_Target": cd_bd.item(),
        "Group_C_MMD_CD_to_Target": mmd_clean,
        "Group_D_MMD_CD_to_Target": mmd_bd,
        "Distribution_ASR_Percent": distribution_asr,
        "Group_C_Utility_CD_to_Source": utility_cd.item(),
    }
    
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
        
    out_path = args.out_json if args.out_json is not None else os.path.join(os.path.dirname(args.ckpt), 'manifold_backdoor_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {out_path}")

if __name__ == '__main__':
    main()
