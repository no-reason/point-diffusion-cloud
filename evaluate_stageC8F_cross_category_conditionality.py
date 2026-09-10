import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import json
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

def compute_cd_numpy_per_sample(pc1, pc2):
    pc1_tensor = torch.tensor(pc1).float()
    pc2_tensor = torch.tensor(pc2).float()
    if pc1_tensor.ndim == 2: pc1_tensor = pc1_tensor.unsqueeze(0)
    if pc2_tensor.ndim == 2: pc2_tensor = pc2_tensor.unsqueeze(0)
    B, N, _ = pc1_tensor.shape
    _, M, _ = pc2_tensor.shape
    cd_list = []
    for b in range(B):
        p1 = pc1_tensor[b].unsqueeze(0)
        p2 = pc2_tensor[0 if M == pc2_tensor.shape[1] and pc2_tensor.shape[0] == 1 else b].unsqueeze(0)
        dist = torch.cdist(p1, p2)
        cd = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
        cd_list.append(cd.item())
    return np.array(cd_list)

def get_target_r(y_target, num_trigger, device, alpha_scale=1.0):
    target_r = torch.zeros_like(y_target)
    r = 1.0
    r_tube = 0.2
    theta = torch.rand(num_trigger) * 2 * np.pi
    phi = torch.rand(num_trigger) * 2 * np.pi
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).to(device)
    target_r[:, -num_trigger:, :] = torus_points * alpha_scale
    return target_r

def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        found = glob.glob(ckpt_path.replace('ckpt_20000.pt', '*/ckpt_20000.pt'))
        if found:
            ckpt_path = max(found, key=os.path.getctime)
        else:
            return None, None
            
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['args']

def plot_point_cloud_2d(ax, pc, title):
    ax.scatter(pc[:, 0], pc[:, 2], s=1, c='b', alpha=0.5)
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=8)
    ax.axis('off')

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_file', type=str, default='./targets/stageC8E_fixed_airplane_target.npy')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k_chair_airplane.h5')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config_name', type=str, required=True)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2: y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)

    clean_model, clean_args = load_model(args.clean_ckpt, device)
    bd_model, bd_args = load_model(args.checkpoint, device)
    if bd_model is None:
        raise ValueError(f"BD model not found at {args.checkpoint}")

    eval_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=['chair'],
        split='test',
        scale_mode='shape_bbox',
    )
    eval_iter = get_data_iterator(DataLoader(eval_dset, batch_size=args.batch_size, num_workers=0, shuffle=False))

    all_traces = {}
    source_pcs = []
    
    num_batches = args.num_samples // args.batch_size
    seed_all(42)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch = next(eval_iter)
            x_source = batch['pointcloud'].to(device)
            source_pcs.append(x_source.cpu().numpy())
            
            X_T_base = torch.randn(args.batch_size, 2048, 3, device=device)
            
            def run_group(model, flex, x_input, group_name):
                z, _ = model.encoder(x_input)
                trace = model.sample(z, 2048, flex, initial_x_T=X_T_base, return_trace=False)
                if group_name not in all_traces:
                    all_traces[group_name] = []
                all_traces[group_name].append(trace.cpu().numpy())

            run_group(clean_model, clean_args.flexibility, x_source, 'A')
            
            target_r_04 = get_target_r(y_target, 200, device, alpha_scale=0.4).expand(args.batch_size, -1, -1)
            x_trig_04 = x_source.clone()
            x_trig_04[:, -200:, :] = target_r_04[:, -200:, :]
            run_group(clean_model, clean_args.flexibility, x_trig_04, 'B')
            
            run_group(bd_model, bd_args.flexibility, x_source, 'C')
            run_group(bd_model, bd_args.flexibility, x_trig_04, 'D')
            
            for scale in [0.2, 0.8]:
                t_r = get_target_r(y_target, 200, device, alpha_scale=scale).expand(args.batch_size, -1, -1)
                x_t = x_source.clone()
                x_t[:, -200:, :] = t_r[:, -200:, :]
                lbl = str(scale).replace('.', '')
                run_group(bd_model, bd_args.flexibility, x_t, f'D{lbl}')

    source_pcs = np.concatenate(source_pcs, axis=0)
    for grp in all_traces:
        all_traces[grp] = np.concatenate(all_traces[grp], axis=0)
        
    records = []
    
    finite_issues = []
    
    for i in range(args.num_samples):
        src = source_pcs[i:i+1]
        
        row = {'sample_id': i, 'config_name': args.config_name}
        
        for grp in ['A', 'B', 'C', 'D', 'D02', 'D08']:
            grp_pc = all_traces[grp][i:i+1]
            if not np.isfinite(grp_pc).all():
                finite_issues.append((args.config_name, grp, i))
                
            cd_src = compute_cd_numpy_per_sample(grp_pc, src)[0]
            cd_tgt = compute_cd_numpy_per_sample(grp_pc, y_target_np)[0]
            
            if grp == 'D':
                row['D04_source_cd'] = cd_src
                row['D04_target_cd'] = cd_tgt
                
            row[f'{grp}_source_cd'] = cd_src
            row[f'{grp}_target_cd'] = cd_tgt
            
        C_s = row['C_source_cd']
        C_t = row['C_target_cd']
        D_s = row['D_source_cd']
        D_t = row['D_target_cd']
        
        clean_preserved = (C_s < C_t)
        attack_success = (D_t < D_s)
        
        row['clean_preserved'] = clean_preserved
        row['attack_success'] = attack_success
        
        row['true_backdoor'] = clean_preserved and attack_success
        row['leakage'] = (not clean_preserved) and attack_success
        row['robust_clean_attack_fail'] = clean_preserved and (not attack_success)
        row['broken'] = (not clean_preserved) and (not attack_success)
        
        row['clean_margin'] = C_t - C_s
        row['attack_margin'] = D_s - D_t
        
        records.append(row)
        
        # Plot up to 8 samples
        if i < 8:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            plot_point_cloud_2d(axs[0], src[0], 'Source Chair')
            plot_point_cloud_2d(axs[1], all_traces['C'][i], f"C Output (Clean Input)\nCD_tgt={C_t:.4f}")
            plot_point_cloud_2d(axs[2], y_target_np[0], 'Fixed Airplane Target')
            plot_point_cloud_2d(axs[3], all_traces['D'][i], f"D Output (Triggered)\nCD_tgt={D_t:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'sample_{i:02d}.png'))
            plt.close(fig)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, 'per_sample_conditionality.csv'), index=False)
    
    summary = {
        'config_name': args.config_name,
        'total_samples': args.num_samples,
        'C_source_cd_mean': df['C_source_cd'].mean(),
        'C_source_cd_median': df['C_source_cd'].median(),
        'C_source_cd_std': df['C_source_cd'].std(),
        'C_target_cd_mean': df['C_target_cd'].mean(),
        'C_target_cd_median': df['C_target_cd'].median(),
        'C_target_cd_std': df['C_target_cd'].std(),
        'D_source_cd_mean': df['D_source_cd'].mean(),
        'D_source_cd_median': df['D_source_cd'].median(),
        'D_source_cd_std': df['D_source_cd'].std(),
        'D_target_cd_mean': df['D_target_cd'].mean(),
        'D_target_cd_median': df['D_target_cd'].median(),
        'D_target_cd_std': df['D_target_cd'].std(),
        
        'D02_source_cd_mean': df['D02_source_cd'].mean(),
        'D02_target_cd_mean': df['D02_target_cd'].mean(),
        'D08_source_cd_mean': df['D08_source_cd'].mean(),
        'D08_target_cd_mean': df['D08_target_cd'].mean(),
        
        'TrueBackdoorRate': df['true_backdoor'].mean(),
        'LeakageRate': df['leakage'].mean(),
        'RobustCleanAttackFailRate': df['robust_clean_attack_fail'].mean(),
        'BrokenRate': df['broken'].mean(),
        'CleanPreservationRate': df['clean_preserved'].mean(),
        'AttackSuccessRate': df['attack_success'].mean(),
        
        'mean_clean_margin': df['clean_margin'].mean(),
        'mean_attack_margin': df['attack_margin'].mean(),
        
        'paired_success_rate': ((df['clean_margin'] > 0) & (df['attack_margin'] > 0)).mean(),
        
        'finite_issues': len(finite_issues)
    }
    
    with open(os.path.join(args.output_dir, 'summary_conditionality.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Evaluation finished for {args.config_name}. True Backdoor Rate: {summary['TrueBackdoorRate']:.2%}")

if __name__ == '__main__':
    run_evaluation()
