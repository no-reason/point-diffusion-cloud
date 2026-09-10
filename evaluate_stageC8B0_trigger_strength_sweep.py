import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import glob

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

def compute_cd_numpy(pc1, pc2):
    pc1_tensor = torch.tensor(pc1).float()
    pc2_tensor = torch.tensor(pc2).float()
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

def get_target_r(y_target, num_trigger, device, alpha=1.0):
    target_r = torch.zeros_like(y_target)
    r = 1.0
    r_tube = 0.2 * alpha
    theta = torch.rand(num_trigger) * 2 * np.pi
    phi = torch.rand(num_trigger) * 2 * np.pi
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).to(device)
    target_r[:, -num_trigger:, :] = torus_points * 0.2 * alpha
    return target_r

def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        if "*" in ckpt_path:
            found = glob.glob(ckpt_path)
        else:
            found = glob.glob(ckpt_path.replace('ckpt_10000.pt', '*/ckpt_10000.pt'))
        if found:
            ckpt_path = max(found, key=os.path.getctime)
        else:
            return None, None
            
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['args']

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--c5_ckpt', type=str, default='./logs_stageC/StageC5_SourceZ_SGPilot/ckpt_10000.pt')
    parser.add_argument('--c6_ckpt', type=str, default='./logs_stageC/StageC6_VAEMediatedInputTrigger_FixedChair_Pilot*/ckpt_10000.pt')
    parser.add_argument('--c7_ckpt', type=str, default='./logs_stageC/StageC7_DualTrigger_FixedChair_Pilot*/ckpt_10000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='./results_stageC8B0_trigger_strength_sweep')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2:
        y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)

    eval_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=['chair'],
        split='test',
        scale_mode='shape_bbox',
    )
    
    alpha_list = [1, 2, 4, 8, 16]
    results = {}
    csv_lines = ["model,alpha,group,cd_target_mean,cd_source_mean"]
    
    def evaluate_group(model, z, initial_x_T, flexibility, name, alpha, group_label):
        trace = model.sample(z, 2048, flexibility, initial_x_T=initial_x_T, return_trace=False)
        out_np = trace.cpu().numpy()
        cd_t = compute_cd_numpy(out_np, y_target_np).mean()
        cd_s = compute_cd_numpy(out_np, x_source_np).mean()
        if name not in results: results[name] = {}
        if alpha not in results[name]: results[name][alpha] = {}
        results[name][alpha][group_label] = {'cd_target': cd_t, 'cd_source': cd_s}
        csv_lines.append(f"{name},{alpha},{group_label},{cd_t:.4f},{cd_s:.4f}")
        return cd_t, cd_s

    models = {}
    for name, path in [('clean', args.clean_ckpt), ('C5', args.c5_ckpt), ('C6', args.c6_ckpt), ('C7', args.c7_ckpt)]:
        m, m_args = load_model(path, device)
        if m is not None:
            models[name] = (m, m_args.flexibility)
            
    if 'clean' not in models:
        print("Clean model not found. Exiting.")
        return

    num_batches = args.num_samples // args.batch_size
    seed_all(42)

    # We evaluate sequentially to avoid OOM
    for alpha in alpha_list:
        print(f"Evaluating Alpha = {alpha}")
        eval_iter = get_data_iterator(DataLoader(eval_dset, batch_size=args.batch_size, num_workers=0, shuffle=False))
        
        for i in range(num_batches):
            batch = next(eval_iter)
            x_source = batch['pointcloud'].to(device)
            x_source_np = x_source.cpu().numpy()
            
            target_r_alpha = get_target_r(y_target, 200, device, alpha=alpha).expand(args.batch_size, -1, -1)
            x_trig_alpha = x_source.clone()
            x_trig_alpha[:, -200:, :] = target_r_alpha[:, -200:, :]
            
            X_T_base = torch.randn(args.batch_size, 2048, 3, device=device)
            X_T_trig_alpha = X_T_base + target_r_alpha
            
            # Clean model leakage baseline
            m, flex = models['clean']
            z_clean, _ = m.encoder(x_source)
            evaluate_group(m, z_clean, X_T_trig_alpha, flex, 'clean', alpha, 'Clean Leakage (X_T+a*r)')
            
            if 'C5' in models:
                m, flex = models['C5']
                z_c5, _ = m.encoder(x_source)
                if alpha == 1: evaluate_group(m, z_c5, X_T_base, flex, 'C5', alpha, 'Normal')
                evaluate_group(m, z_c5, X_T_trig_alpha, flex, 'C5', alpha, 'Triggered (X_T+a*r)')
                
            if 'C6' in models:
                m, flex = models['C6']
                z_c6_trig, _ = m.encoder(x_trig_alpha)
                if alpha == 1: 
                    z_c6, _ = m.encoder(x_source)
                    evaluate_group(m, z_c6, X_T_base, flex, 'C6', alpha, 'Normal x')
                evaluate_group(m, z_c6_trig, X_T_base, flex, 'C6', alpha, 'Triggered x')
                
            if 'C7' in models:
                m, flex = models['C7']
                z_c7, _ = m.encoder(x_source)
                z_c7_trig, _ = m.encoder(x_trig_alpha)
                if alpha == 1: evaluate_group(m, z_c7, X_T_base, flex, 'C7', alpha, 'C (Normal)')
                evaluate_group(m, z_c7, X_T_trig_alpha, flex, 'C7', alpha, 'D (Diff Trig)')
                evaluate_group(m, z_c7_trig, X_T_base, flex, 'C7', alpha, 'E (Input Trig)')
                evaluate_group(m, z_c7_trig, X_T_trig_alpha, flex, 'C7', alpha, 'F (Dual Trig)')

    with open(os.path.join(args.out_dir, 'metrics_trigger_strength_sweep.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(os.path.join(args.out_dir, 'metrics_trigger_strength_sweep.csv'), 'w') as f:
        f.write('\n'.join(csv_lines))

    print("Trigger Strength Sweep finished.")

if __name__ == '__main__':
    run_evaluation()
