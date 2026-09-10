import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import json
import argparse
import glob
import matplotlib.pyplot as plt

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
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2: y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)
    
    chair_target_np = np.load("targets/stage3_fixed_chair_target.npy")
    if chair_target_np.ndim == 2: chair_target_np = chair_target_np[np.newaxis, ...]

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

    results = {}
    csv_lines = ["group,cd_target_mean,cd_source_mean,cd_chair_target_mean"]
    
    def get_stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75))
        }

    all_traces = {}
    source_pcs = []
    trig_source_pcs = {}
    
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
            if '0.4' not in trig_source_pcs: trig_source_pcs['0.4'] = []
            trig_source_pcs['0.4'].append(x_trig_04.cpu().numpy())
            
            run_group(clean_model, clean_args.flexibility, x_trig_04, 'B')
            run_group(bd_model, bd_args.flexibility, x_source, 'C')
            run_group(bd_model, bd_args.flexibility, x_trig_04, 'D')
            
            for scale in [0.2, 0.4, 0.8]:
                t_r = get_target_r(y_target, 200, device, alpha_scale=scale).expand(args.batch_size, -1, -1)
                x_t = x_source.clone()
                x_t[:, -200:, :] = t_r[:, -200:, :]
                lbl = str(scale).replace('.', '')
                run_group(bd_model, bd_args.flexibility, x_t, f'D{lbl}')
                
                if str(scale) not in trig_source_pcs: trig_source_pcs[str(scale)] = []
                trig_source_pcs[str(scale)].append(x_t.cpu().numpy())

    source_pcs = np.concatenate(source_pcs, axis=0)
    for k in trig_source_pcs:
        trig_source_pcs[k] = np.concatenate(trig_source_pcs[k], axis=0)
        
    for grp, traces in all_traces.items():
        traces = np.concatenate(traces, axis=0)
        all_traces[grp] = traces
        
        cd_t = compute_cd_numpy(traces, y_target_np)
        cd_s = compute_cd_numpy(traces, source_pcs)
        cd_ct = compute_cd_numpy(traces, chair_target_np)
        
        results[grp] = {
            'cd_target': get_stats(cd_t),
            'cd_source': get_stats(cd_s),
            'cd_chair_target': get_stats(cd_ct)
        }
        csv_lines.append(f"{grp},{cd_t.mean():.4f},{cd_s.mean():.4f},{cd_ct.mean():.4f}")

    results['attack_gain'] = results['C']['cd_target']['mean'] - results['D']['cd_target']['mean']
    results['trigger_effect'] = results['D']['cd_source']['mean'] - results['C']['cd_source']['mean']

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(os.path.join(args.output_dir, 'samples.csv'), 'w') as f:
        f.write('\n'.join(csv_lines))

    # Save 8 sample visualizations
    for i in range(min(8, args.num_samples)):
        for grp in ['A', 'B', 'C', 'D', 'D02', 'D04', 'D08']:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            plot_point_cloud_2d(axs[0], source_pcs[i], 'Source Chair')
            
            # Identify triggered input
            if grp in ['B', 'D', 'D04']:
                t_input = trig_source_pcs['0.4'][i]
            elif grp == 'D02':
                t_input = trig_source_pcs['0.2'][i]
            elif grp == 'D08':
                t_input = trig_source_pcs['0.8'][i]
            else:
                t_input = source_pcs[i] # A and C use clean
                
            plot_point_cloud_2d(axs[1], t_input, 'Conditioning Input')
            plot_point_cloud_2d(axs[2], y_target_np[0], 'Fixed Airplane Target')
            
            cd_val = compute_cd_numpy(all_traces[grp][i:i+1], y_target_np)[0]
            plot_point_cloud_2d(axs[3], all_traces[grp][i], f"Output (Group {grp})\nCD to target: {cd_val:.4f}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'sample_{i:02d}_{grp}.png'))
            plt.close(fig)

    print(f"Evaluation finished.")

if __name__ == '__main__':
    run_evaluation()
