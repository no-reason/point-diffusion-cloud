import os
import argparse
import torch
import numpy as np
import copy
from tqdm import tqdm
import csv

from models.vae_gaussian import GaussianVAE
from utils.misc import *
from tools.input_triggers import apply_input_trigger

def chamfer_distance(p1, p2):
    ''' p1: (1, N, 3), p2: (1, N, 3) '''
    diff = p1.unsqueeze(2) - p2.unsqueeze(1) # (1, N, N, 3)
    dist2 = torch.sum(diff ** 2, dim=-1) # (1, N, N)
    d1 = torch.min(dist2, dim=2)[0].mean()
    d2 = torch.min(dist2, dim=1)[0].mean()
    return (d1 + d2) / 2

def cosine_similarity(v1, v2):
    v1_norm = torch.nn.functional.normalize(v1, p=2, dim=1)
    v2_norm = torch.nn.functional.normalize(v2, p=2, dim=1)
    return torch.sum(v1_norm * v2_norm, dim=1)

def optimize_delta(x, y_target, encoder, trigger_fn, args, M_mask, N_K):
    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    with torch.no_grad():
        z_x, _ = encoder(x)
        z_x = z_x.detach()
        z_y, _ = encoder(y_target)
        z_y = z_y.detach()
        v_target = z_y - z_x
        
        x_g = trigger_fn(x)
        z_g, _ = encoder(x_g)
        z_g = z_g.detach()
        v_trigger_before = z_g - z_x

    logs = {'initial': None, 'final': None}
    
    for step in range(args.steps):
        optimizer.zero_grad()
        
        x_delta = x + delta * M_mask
        z_delta, _ = encoder(x_delta)
        
        x_delta_g = trigger_fn(x_delta)
        z_delta_g, _ = encoder(x_delta_g)

        v_delta = z_delta - z_x
        v_trigger_after = z_delta_g - z_delta

        align_loss = args.lambda_align * (1 - cosine_similarity(v_trigger_after, v_target).mean())
        dist_loss = args.lambda_dist * torch.norm(z_delta_g - z_y, p=2, dim=1).mean()
        clean_loss = args.lambda_clean * torch.norm(z_delta - z_x, p=2, dim=1).mean()
        
        orth_loss = args.lambda_orth * (
            cosine_similarity(v_delta, v_target).pow(2).mean() +
            cosine_similarity(v_delta, v_trigger_after).pow(2).mean()
        )

        geo_loss = args.lambda_geo * (chamfer_distance(x_delta, x) + args.alpha * torch.norm(delta * M_mask, p=2))

        loss = align_loss + dist_loss + clean_loss + orth_loss + geo_loss
        loss.backward()
        optimizer.step()

        # Constraints
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -args.epsilon_delta, args.epsilon_delta)
            delta.data *= M_mask
            delta.data[:, N_K:, :] = 0

        if step == 0:
            logs['initial'] = {
                'total_loss': loss.item(),
                'align_loss': align_loss.item(),
                'orth_loss': orth_loss.item(),
                'geo_loss': geo_loss.item()
            }
            
    logs['final'] = {
        'total_loss': loss.item(),
        'align_loss': align_loss.item(),
        'orth_loss': orth_loss.item(),
        'geo_loss': geo_loss.item()
    }
    
    return delta.detach(), logs

def get_failed_and_success_sources(args):
    failed = ["007", "013", "dataset_26", "dataset_29", "dataset_34"]
    success = ["001", "002", "003", "004", "005"]
    
    from glob import glob
    source_paths = []
    labels = []
    
    for f in failed[:args.num_failed]:
        matches = glob(f"/data/personal_data/zyy/point-diffusion-cloud/results_stageS/StageS2*/visualizations/*_{f}_*.npy")
        if not matches:
             matches = glob(f"/data/personal_data/zyy/point-diffusion-cloud/results_stageC/StageC9*/visualizations/*_{f}_*.npy")
        if matches:
            source_paths.append(matches[0])
            labels.append(f"failed_{f}")
            
    for s in success[:args.num_success]:
        matches = glob(f"/data/personal_data/zyy/point-diffusion-cloud/results_stageS/StageS2*/visualizations/*_{s}_*.npy")
        if not matches:
            matches = glob(f"/data/personal_data/zyy/point-diffusion-cloud/results_stageC/StageC9*/visualizations/*_{s}_*.npy")
        if matches:
            source_paths.append(matches[0])
            labels.append(f"success_{s}")

    return source_paths, labels

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 1. Load VAE
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    vae_args = ckpt['args']
    model = GaussianVAE(vae_args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(args.device)
    model.eval()
    encoder = model.encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # 2. Load Target
    y_target = np.load(args.target_path)
    y_target_tensor = torch.from_numpy(y_target).float().unsqueeze(0).to(args.device)

    # 3. Trigger configuration
    N = 2048
    K = args.n_trigger
    M_mask = torch.zeros(1, N, 3).to(args.device)
    M_mask[:, :N-K, :] = 1.0
    
    def trigger_fn(points):
        return apply_input_trigger(
            points, 
            trigger_type=args.trigger_type, 
            n_trigger=args.n_trigger,
            trigger_scale=args.trigger_scale,
            center=args.trigger_center
        )

    # 4. Sources
    source_paths, labels = get_failed_and_success_sources(args)
    if len(source_paths) == 0:
         print("WARNING: Could not auto-detect sources. Loading randomly from h5")
         import h5py
         f = h5py.File('/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k_chair_airplane.h5', 'r')
         chairs = f['03001627']['test']
         for i in range(args.num_failed + args.num_success):
             source_paths.append(chairs[i])
             labels.append(f"h5_chair_{i}")
             
    csv_file = open(os.path.join(args.output_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    header = ['source_id', 'group', 'CD_xdelta_x', 'delta_l2', 'delta_linf', 'delta_mean_abs', 'delta_last_K_max_abs', 'finite_ratio_xdelta',
              'dist_xdelta_x', 'dist_Tg_x_target_before', 'dist_Tg_xdelta_target_after',
              'cos_delta_target', 'cos_trigger_target_before', 'cos_trigger_target_after', 'cos_delta_trigger',
              'target_dist_gain', 'cos_trigger_target_gain', 'init_total_loss', 'final_total_loss']
    writer.writerow(header)

    for src, label in zip(source_paths, labels):
        print(f"Processing {label}...")
        if isinstance(src, str) and src.endswith('.npy'):
            x_np = np.load(src)
        else:
            x_np = src
            
        if x_np.shape != (2048, 3):
            continue
            
        x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).to(args.device)
        
        delta_opt, logs = optimize_delta(x_tensor, y_target_tensor, encoder, trigger_fn, args, M_mask, N-K)
        
        x_delta = x_tensor + delta_opt * M_mask
        
        with torch.no_grad():
            z_x, _ = encoder(x_tensor)
            z_y, _ = encoder(y_target_tensor)
            z_xdelta, _ = encoder(x_delta)
            
            x_g = trigger_fn(x_tensor)
            z_g, _ = encoder(x_g)
            
            xdelta_g = trigger_fn(x_delta)
            zxdelta_g, _ = encoder(xdelta_g)
            
            # Metrics
            cd = chamfer_distance(x_delta, x_tensor).item()
            dl2 = torch.norm(delta_opt * M_mask, p=2).item()
            dlinf = torch.abs(delta_opt * M_mask).max().item()
            dmean = torch.abs(delta_opt * M_mask).mean().item()
            dlastK = torch.abs(delta_opt[:, N-K:, :]).max().item()
            fin = torch.isfinite(x_delta).float().mean().item()
            
            dist_xdelta_x = torch.norm(z_xdelta - z_x, p=2).item()
            dist_Tg_x_target = torch.norm(z_g - z_y, p=2).item()
            dist_Tg_xdelta_target = torch.norm(zxdelta_g - z_y, p=2).item()
            
            cos_delta_tgt = cosine_similarity(z_xdelta - z_x, z_y - z_x).item()
            cos_trig_tgt_bef = cosine_similarity(z_g - z_x, z_y - z_x).item()
            cos_trig_tgt_aft = cosine_similarity(zxdelta_g - z_xdelta, z_y - z_x).item()
            cos_del_trig = cosine_similarity(z_xdelta - z_x, zxdelta_g - z_xdelta).item()
            
        writer.writerow([
            label, label.split('_')[0], cd, dl2, dlinf, dmean, dlastK, fin,
            dist_xdelta_x, dist_Tg_x_target, dist_Tg_xdelta_target,
            cos_delta_tgt, cos_trig_tgt_bef, cos_trig_tgt_aft, cos_del_trig,
            dist_Tg_x_target - dist_Tg_xdelta_target, cos_trig_tgt_aft - cos_trig_tgt_bef,
            logs['initial']['total_loss'], logs['final']['total_loss']
        ])
        
        np.save(os.path.join(args.output_dir, f"{label}_delta.npy"), (delta_opt*M_mask).cpu().numpy())
        
        try:
             import matplotlib.pyplot as plt
             fig = plt.figure(figsize=(10, 10))
             
             pts = [x_np, x_delta[0].cpu().numpy(), x_g[0].cpu().numpy(), xdelta_g[0].cpu().numpy()]
             titles = ['Clean Source (x)', 'x + delta', 'Tg(x)', 'Tg(x + delta)']
             
             for i in range(4):
                 ax = fig.add_subplot(2, 2, i+1, projection='3d')
                 ax.scatter(pts[i][:,0], pts[i][:,1], pts[i][:,2], s=2, c='b')
                 if i >= 2:
                     ax.scatter(pts[i][N-K:,0], pts[i][N-K:,1], pts[i][N-K:,2], s=10, c='r')
                 ax.set_title(titles[i])
                 ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
             
             plt.savefig(os.path.join(args.output_dir, 'visualizations', f'source_{label}_quad.png'))
             plt.close(fig)
        except Exception as e:
             print("Plotting error:", e)

    csv_file.close()
    print("Pilot completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--trigger_type', type=str, default='small_sphere')
    parser.add_argument('--n_trigger', type=int, default=200)
    parser.add_argument('--trigger_scale', type=float, default=0.05)
    parser.add_argument('--trigger_center', nargs='+', type=float, default=[0.9, -0.9, -0.9])
    
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epsilon_delta', type=float, default=0.02)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_dist', type=float, default=0.1)
    parser.add_argument('--lambda_clean', type=float, default=1.0)
    parser.add_argument('--lambda_orth', type=float, default=0.2)
    parser.add_argument('--lambda_geo', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    
    parser.add_argument('--num_failed', type=int, default=5)
    parser.add_argument('--num_success', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
