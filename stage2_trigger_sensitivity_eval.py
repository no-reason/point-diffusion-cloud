import os
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("/data/personal_data/zyy/point-diffusion-cloud")

from utils.dataset import ShapeNetCore
from models.vae_gaussian_bd import GaussianVAE
from tools.input_triggers import apply_input_trigger
from chamferdist import ChamferDistance

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results_stage2_trigger_sensitivity_smoke')
    parser.add_argument('--ckpt', type=str, default='/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--dataset_path', type=str, default='/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k.h5')
    parser.add_argument('--category', type=str, default='chair')
    return parser.parse_args()

def normalize_pc(pc):
    pc_max = pc.max(dim=0, keepdim=True)[0]
    pc_min = pc.min(dim=0, keepdim=True)[0]
    shift = ((pc_min + pc_max) / 2).view(1, 3)
    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
    return (pc - shift) / scale

def batch_normalize_pc(pcs):
    for i in range(pcs.size(0)):
        pcs[i] = normalize_pc(pcs[i])
    return pcs

def generate_local_trigger(x, trigger_type, n_trigger, trigger_scale, center):
    B, N, _ = x.shape
    K = n_trigger
    device = x.device
    
    if trigger_type == "torus" or trigger_type == "ring":
        x_trigger, _ = apply_input_trigger(
            x, trigger_type=trigger_type, n_trigger=K, trigger_scale=trigger_scale, 
            trigger_position="fixed_global", center=center, return_info=True, shuffle=False
        )
        return x_trigger
    elif trigger_type == "large_torus":
        x_trigger, _ = apply_input_trigger(
            x, trigger_type="torus", n_trigger=K, trigger_scale=trigger_scale, 
            trigger_position="fixed_global", center=center, return_info=True, shuffle=False
        )
        return x_trigger
    elif trigger_type == "fixed_global_cluster" or trigger_type == "random_cluster":
        # Fixed template
        torch.manual_seed(0)
        template = (torch.rand(1, K, 3, device=device) * 2 - 1) * trigger_scale
        template = template + torch.tensor(center, device=device).view(1, 1, 3)
        trigger_points = template.repeat(B, 1, 1)
        x_keep = x[:, :N-K, :]
        x_trigger = torch.cat([x_keep, trigger_points], dim=1)
        return x_trigger
    else:
        raise ValueError(f"Unknown trigger {trigger_type}")

def train_linear_probe(z_clean, z_triggered):
    # z_clean, z_triggered: [N, D]
    import torch.nn as nn
    import torch.optim as optim
    
    with torch.enable_grad():
        N = z_clean.size(0)
        if N < 4:
            return 0.5 # Too small
        
        X = torch.cat([z_clean, z_triggered], dim=0)
        Y = torch.cat([torch.zeros(N), torch.ones(N)], dim=0).long().to(X.device)
        
        # Shuffle and split 70/30
        perm = torch.randperm(2 * N)
        X, Y = X[perm], Y[perm]
        split = int(0.7 * 2 * N)
        
        X_train, Y_train = X[:split], Y[:split]
        X_test, Y_test = X[split:], Y[split:]
        
        if len(X_train) == 0 or len(X_test) == 0:
            return 0.5
            
        model = nn.Linear(X.size(1), 2).to(X.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(100):
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, Y_train)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            out = model(X_test)
            pred = out.argmax(dim=1)
            acc = (pred == Y_test).float().mean().item()
            
        return acc

def plot_point_clouds(clean, triggered, clean_gen, triggered_gen, fixed_target, earphone_target, path):
    fig = plt.figure(figsize=(20, 4))
    
    pcs = [clean, triggered, clean_gen, triggered_gen, fixed_target, earphone_target]
    titles = ["Clean Input", "Triggered Input", "Clean Gen", "Triggered Gen", "Fixed Target", "Earphone Target"]
    
    for i, (pc, title) in enumerate(zip(pcs, titles)):
        ax = fig.add_subplot(1, 6, i+1, projection='3d')
        if pc is not None:
            pc_np = pc.detach().cpu().numpy()
            ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=1, c=pc_np[:, 2], cmap='viridis')
        ax.set_title(title)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def main():
    args = get_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples_npy"), exist_ok=True)
    
    device = torch.device(args.device)
    
    print("Loading dataset...")
    dset = ShapeNetCore(path=args.dataset_path, cates=[args.category], split='test', scale_mode='shape_bbox')
    
    print("Loading model...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model_args = ckpt['args']
    model = GaussianVAE(model_args).to(device)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    print(f"Loaded clean model. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    chamfer_dist = ChamferDistance().to(device)
    
    # Load targets
    # 1. Fixed chair target
    fixed_chair_target = normalize_pc(dset[0]['pointcloud'].to(device))
    np.save(os.path.join(args.output_dir, "samples_npy", "fixed_chair_target.npy"), fixed_chair_target.cpu().numpy())
    
    # 2. Earphone target
    earphone_path = "/data/personal_data/zyy/point-diffusion-cloud/target_earphone.npy"
    earphone_target = None
    if os.path.exists(earphone_path):
        earphone_target = torch.from_numpy(np.load(earphone_path)).float().to(device)
        if len(earphone_target.shape) == 2:
            earphone_target = earphone_target.unsqueeze(0)
        # Normalize to shape_bbox
        earphone_target = normalize_pc(earphone_target[0])
        np.save(os.path.join(args.output_dir, "samples_npy", "earphone_target.npy"), earphone_target.cpu().numpy())
    
    triggers_config = {
        "torus": {"n_trigger": 128, "trigger_scale": 0.10, "center": (0.6, 0.6, 0.6)},
        "large_torus": {"n_trigger": 256, "trigger_scale": 0.15, "center": (0.6, 0.6, 0.6)},
        "ring": {"n_trigger": 128, "trigger_scale": 0.10, "center": (0.6, 0.6, 0.6)},
        "fixed_global_cluster": {"n_trigger": 128, "trigger_scale": 0.10, "center": (0.6, 0.6, 0.6)},
        "random_cluster": {"n_trigger": 128, "trigger_scale": 0.10, "center": (0.6, 0.6, 0.6)}
    }
    
    metrics = {
        "checkpoint": args.ckpt,
        "category": args.category,
        "num_eval": args.num_eval,
        "model_class_used": "GaussianVAE",
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "use_encoder_mean": True,
        "triggers": {}
    }
    
    per_trigger_latent_metrics = []
    per_sample_metrics_list = []
    
    # Gather evaluation data
    N_eval = min(args.num_eval, len(dset))
    all_x = []
    for i in range(N_eval):
        all_x.append(dset[i]['pointcloud'])
    all_x = torch.stack(all_x).to(device)
    all_x = batch_normalize_pc(all_x)
    
    with torch.no_grad():
        # Get baseline clean latents and outputs
        z_clean_list = []
        x_clean_gen_list = []
        
        for i in tqdm(range(0, N_eval, args.batch_size), desc="Clean Baseline"):
            batch_x = all_x[i:i+args.batch_size]
            
            # Encoder mean
            z_mu, _ = model.encoder(batch_x)
            z_clean_list.append(z_mu)
            
            # Generate (fixed seed for stable sampling)
            torch.manual_seed(42)
            gen = model.sample(z_mu, num_points=2048, flexibility=model_args.flexibility)
            x_clean_gen_list.append(gen)
            
        z_clean = torch.cat(z_clean_list, dim=0)
        x_clean_gen = torch.cat(x_clean_gen_list, dim=0)
        x_clean_gen = batch_normalize_pc(x_clean_gen)
        
        # Calculate base distances
        cd_clean_to_input = chamfer_dist(x_clean_gen, all_x, bidirectional=True, batch_reduction=None, point_reduction='mean')
        cd_clean_to_fixed = chamfer_dist(x_clean_gen, fixed_chair_target.unsqueeze(0).expand(N_eval, -1, -1), bidirectional=True, batch_reduction=None, point_reduction='mean')
        
        cd_clean_to_earphone = None
        if earphone_target is not None:
            cd_clean_to_earphone = chamfer_dist(x_clean_gen, earphone_target.unsqueeze(0).expand(N_eval, -1, -1), bidirectional=True, batch_reduction=None, point_reduction='mean')
    
    for trigger_name, t_cfg in triggers_config.items():
        print(f"\nEvaluating {trigger_name}...")
        
        # Apply trigger
        x_triggered = generate_local_trigger(all_x, trigger_name, t_cfg["n_trigger"], t_cfg["trigger_scale"], t_cfg["center"])
        
        with torch.no_grad():
            z_triggered_list = []
            x_triggered_gen_list = []
            
            for i in tqdm(range(0, N_eval, args.batch_size), desc=f"Triggered ({trigger_name})"):
                batch_x_trig = x_triggered[i:i+args.batch_size]
                
                z_mu, _ = model.encoder(batch_x_trig)
                z_triggered_list.append(z_mu)
                
                # Generate with same seed to cancel noise
                torch.manual_seed(42)
                gen = model.sample(z_mu, num_points=2048, flexibility=model_args.flexibility)
                x_triggered_gen_list.append(gen)
                
            z_triggered = torch.cat(z_triggered_list, dim=0)
            x_triggered_gen = torch.cat(x_triggered_gen_list, dim=0)
            x_triggered_gen = batch_normalize_pc(x_triggered_gen)
            
            # Stage 2A: Latent Sensitivity
            delta_z = z_triggered - z_clean
            
            z_norm = torch.norm(z_clean, p=2, dim=1)
            z_g_norm = torch.norm(z_triggered, p=2, dim=1)
            delta_z_l2 = torch.norm(delta_z, p=2, dim=1)
            rel_delta_norm = delta_z_l2 / (z_norm + 1e-8)
            
            # Pairwise cosine similarity of delta_z
            delta_z_normalized = delta_z / (delta_z_l2.unsqueeze(1) + 1e-8)
            cos_sim_matrix = torch.mm(delta_z_normalized, delta_z_normalized.t())
            
            # upper triangle without diagonal
            idx = torch.triu_indices(N_eval, N_eval, offset=1)
            pairwise_cosines = cos_sim_matrix[idx[0], idx[1]]
            
            # Linear probe
            probe_acc = train_linear_probe(z_clean, z_triggered)
            
            # Stage 2B: Output Sensitivity
            cd_trigger_to_input = chamfer_dist(x_triggered_gen, all_x, bidirectional=True, batch_reduction=None, point_reduction='mean')
            cd_trigger_to_clean = chamfer_dist(x_triggered_gen, x_clean_gen, bidirectional=True, batch_reduction=None, point_reduction='mean')
            cd_trigger_to_fixed = chamfer_dist(x_triggered_gen, fixed_chair_target.unsqueeze(0).expand(N_eval, -1, -1), bidirectional=True, batch_reduction=None, point_reduction='mean')
            
            fixed_gain = cd_clean_to_fixed - cd_trigger_to_fixed
            
            earphone_gain = None
            if cd_clean_to_earphone is not None:
                cd_trigger_to_earphone = chamfer_dist(x_triggered_gen, earphone_target.unsqueeze(0).expand(N_eval, -1, -1), bidirectional=True, batch_reduction=None, point_reduction='mean')
                earphone_gain = cd_clean_to_earphone - cd_trigger_to_earphone
                cd_trigger_to_earphone_mean = cd_trigger_to_earphone.mean().item()
                earphone_gain_mean = earphone_gain.mean().item()
            else:
                cd_trigger_to_earphone_mean = None
                earphone_gain_mean = None
                
            finite_ratio_clean = torch.isfinite(x_clean_gen).all(dim=2).all(dim=1).float().mean().item()
            finite_ratio_trigger = torch.isfinite(x_triggered_gen).all(dim=2).all(dim=1).float().mean().item()
            
            # Save visualizations
            for i in range(min(16, N_eval)):
                path = os.path.join(args.output_dir, "visualizations", f"trigger_{trigger_name}_sample_{i:03d}.png")
                plot_point_clouds(
                    all_x[i], x_triggered[i], x_clean_gen[i], x_triggered_gen[i], 
                    fixed_chair_target, earphone_target, path
                )
            
            # Metrics dict
            tm = {
                "delta_z_l2_mean": delta_z_l2.mean().item(),
                "relative_delta_norm_mean": rel_delta_norm.mean().item(),
                "pairwise_delta_cosine_mean": pairwise_cosines.mean().item(),
                "linear_probe_accuracy": probe_acc,
                "CD_trigger_to_clean_mean": cd_trigger_to_clean.mean().item(),
                "CD_clean_to_fixed_target_mean": cd_clean_to_fixed.mean().item(),
                "CD_trigger_to_fixed_target_mean": cd_trigger_to_fixed.mean().item(),
                "fixed_target_gain_mean": fixed_gain.mean().item(),
                "finite_ratio_trigger_gen": finite_ratio_trigger,
                "earphone_gain_mean": earphone_gain_mean
            }
            metrics["triggers"][trigger_name] = tm
            
            tm["trigger"] = trigger_name
            per_trigger_latent_metrics.append(tm)
            
            # Save per-sample metrics
            for i in range(N_eval):
                viz_path = os.path.join(args.output_dir, "visualizations", f"trigger_{trigger_name}_sample_{i:03d}.png") if i < 16 else ""
                
                finite = torch.isfinite(x_triggered_gen[i]).all().item()
                
                per_sample_metrics_list.append({
                    "sample_id": i,
                    "trigger": trigger_name,
                    "delta_z_l2": delta_z_l2[i].item(),
                    "relative_delta_norm": rel_delta_norm[i].item(),
                    "CD_trigger_to_clean": cd_trigger_to_clean[i].item(),
                    "CD_clean_to_fixed_target": cd_clean_to_fixed[i].item(),
                    "CD_trigger_to_fixed_target": cd_trigger_to_fixed[i].item(),
                    "fixed_target_gain": fixed_gain[i].item(),
                    "CD_clean_to_earphone": cd_clean_to_earphone[i].item() if cd_clean_to_earphone is not None else None,
                    "CD_trigger_to_earphone": cd_trigger_to_earphone[i].item() if earphone_gain is not None else None,
                    "earphone_gain": earphone_gain[i].item() if earphone_gain is not None else None,
                    "finite_trigger": finite,
                    "visualization_path": viz_path
                })

    # Save JSON
    with open(os.path.join(args.output_dir, "metrics_stage2_trigger_sensitivity.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save CSVs
    df_trigger = pd.DataFrame(per_trigger_latent_metrics)
    df_trigger.to_csv(os.path.join(args.output_dir, "per_trigger_latent_metrics.csv"), index=False)
    
    df_sample = pd.DataFrame(per_sample_metrics_list)
    df_sample.to_csv(os.path.join(args.output_dir, "per_sample_metrics.csv"), index=False)
    
    print("Stage 2 Evaluation Completed!")

if __name__ == "__main__":
    main()
