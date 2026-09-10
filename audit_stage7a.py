import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm

from utils.dataset import ShapeNetCore
from models.vae_gaussian import GaussianVAE
from utils.misc import seed_all
from verify_stage7a import compute_cd_pytorch, plot_point_clouds_grid

def part1_audit_h5(h5_path, out_dir):
    res = {}
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        res['keys'] = keys
        res['shapes'] = {}
        for k in keys:
            res['shapes'][k] = {}
            for split in ['train', 'val', 'test']:
                if split in f[k]:
                    res['shapes'][k][split] = f[k][split].shape
                else:
                    res['shapes'][k][split] = None
                    
    # utils.dataset mapping
    res['label_to_category'] = {
        '02691156': 'airplane',
        '03001627': 'chair'
    }
    res['chair_label'] = '03001627'
    res['airplane_label'] = '02691156'
    
    with open(os.path.join(out_dir, 'audit_h5_labels.json'), 'w') as f:
        json.dump(res, f, indent=4)
        
    return res

def part2_raw_visualization(h5_path, out_dir):
    vis_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        chair_raw = f['03001627']['test'][:8]
        airplane_raw = f['02691156']['test'][:8]
        
    def plot_8(pcs, title_prefix, filename):
        fig = plt.figure(figsize=(16, 8))
        for i in range(8):
            ax = fig.add_subplot(2, 4, i + 1, projection='3d')
            pc = pcs[i]
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='.')
            ax.set_title(f"{title_prefix} {i}")
            ax.set_axis_off()
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-0.5, 0.5)
            ax.view_init(elev=20, azim=30)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, filename))
        plt.close()
        
    plot_8(chair_raw, "Raw Chair", 'raw_chair_inputs_grid.png')
    plot_8(airplane_raw, "Raw Airplane", 'raw_airplane_inputs_grid.png')
    
    fig = plt.figure(figsize=(16, 8))
    for i in range(4):
        # chair
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        pc = chair_raw[i]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='.')
        ax.set_title(f"Chair {i}")
        ax.set_axis_off()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.view_init(elev=20, azim=30)
        
        # airplane
        ax = fig.add_subplot(2, 4, i + 5, projection='3d')
        pc = airplane_raw[i]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='r', marker='.')
        ax.set_title(f"Airplane {i}")
        ax.set_axis_off()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'raw_chair_airplane_side_by_side.png'))
    plt.close()

def part4_audit_checkpoint(ckpt_path, out_dir, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = GaussianVAE(ckpt['args']).to(device)
    load_res = model.load_state_dict(ckpt['state_dict'], strict=False)
    
    res = {
        "checkpoint_path": ckpt_path,
        "missing_keys_count": len(load_res.missing_keys),
        "unexpected_keys_count": len(load_res.unexpected_keys),
        "missing_keys": load_res.missing_keys,
        "unexpected_keys": load_res.unexpected_keys,
        "strict_load_used": False,
        "checkpoint_args": vars(ckpt['args'])
    }
    with open(os.path.join(out_dir, 'audit_checkpoint_loading.json'), 'w') as f:
        json.dump(res, f, indent=4)
        
    return model

def part5_latent_separability(model, chair_x, airplane_x, out_dir, device):
    with torch.no_grad():
        z_chair, _ = model.encoder(chair_x.to(device))
        z_air, _ = model.encoder(airplane_x.to(device))
        
    z_chair = z_chair.cpu().numpy()
    z_air = z_air.cpu().numpy()
    
    chair_norm = np.linalg.norm(z_chair, axis=1)
    air_norm = np.linalg.norm(z_air, axis=1)
    
    chair_centroid = z_chair.mean(axis=0)
    air_centroid = z_air.mean(axis=0)
    
    centroid_dist = np.linalg.norm(chair_centroid - air_centroid)
    
    within_chair = np.linalg.norm(z_chair - chair_centroid, axis=1).mean()
    within_air = np.linalg.norm(z_air - air_centroid, axis=1).mean()
    
    X = np.concatenate([z_chair, z_air], axis=0)
    y = np.concatenate([np.zeros(len(z_chair)), np.ones(len(z_air))])
    
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    
    res = {
        "chair_norm_mean": float(chair_norm.mean()),
        "chair_norm_std": float(chair_norm.std()),
        "airplane_norm_mean": float(air_norm.mean()),
        "airplane_norm_std": float(air_norm.std()),
        "chair_airplane_centroid_distance": float(centroid_dist),
        "within_chair_distance": float(within_chair),
        "within_airplane_distance": float(within_air),
        "centroid_to_within_ratio": float(centroid_dist / ((within_chair + within_air)/2)),
        "latent_classification_accuracy": float(scores.mean())
    }
    with open(os.path.join(out_dir, 'latent_separability_stage7a.json'), 'w') as f:
        json.dump(res, f, indent=4)
        
def part6_resample_16(model, chair_16, airplane_16, out_dir, device):
    def get_metrics(x, same_class_x, other_class_x, cat_name):
        x = x.to(device)
        same_class_x = same_class_x.to(device)
        other_class_x = other_class_x.to(device)
        with torch.no_grad():
            z_mu, _ = model.encoder(x)
            seed_all(0)
            x_gen = model.sample(z_mu, 2048, flexibility=0.0, truncate_std=None)
            
        A = compute_cd_pytorch(x_gen, x)
        
        # B
        B_list = []
        nearest_same_idx = []
        for i in range(len(x)):
            valid_idx = [j for j in range(len(same_class_x)) if j != i]
            refs = same_class_x[valid_idx]
            x_gen_i = x_gen[i:i+1].expand(len(valid_idx), -1, -1)
            cds = compute_cd_pytorch(x_gen_i, refs)
            B_list.append(cds.mean().item())
            nearest_same_idx.append(valid_idx[cds.argmin().item()])
            
        # C
        C_list = []
        nearest_other_idx = []
        for i in range(len(x)):
            refs = other_class_x
            x_gen_i = x_gen[i:i+1].expand(len(refs), -1, -1)
            cds = compute_cd_pytorch(x_gen_i, refs)
            C_list.append(cds.mean().item())
            nearest_other_idx.append(cds.argmin().item())
            
        finite = torch.isfinite(x_gen).all(dim=-1).all(dim=-1).cpu().numpy().astype(int)
        
        return x_gen, A.cpu().numpy(), B_list, C_list, nearest_same_idx, nearest_other_idx, finite
        
    c_gen, A_c, B_c, C_c, ns_c, no_c, f_c = get_metrics(chair_16, chair_16, airplane_16, 'chair')
    a_gen, A_a, B_a, C_a, ns_a, no_a, f_a = get_metrics(airplane_16, airplane_16, chair_16, 'airplane')
    
    rows = []
    for i in range(16):
        rows.append({
            'sample_index': i, 'label': '03001627', 'category_name': 'chair',
            'A': A_c[i], 'B': B_c[i], 'C': C_c[i], 
            'nearest_random_same_class_index': ns_c[i], 'nearest_random_other_class_index': no_c[i],
            'finite_ratio': f_c[i]
        })
    for i in range(16):
        rows.append({
            'sample_index': i, 'label': '02691156', 'category_name': 'airplane',
            'A': A_a[i], 'B': B_a[i], 'C': C_a[i], 
            'nearest_random_same_class_index': ns_a[i], 'nearest_random_other_class_index': no_a[i],
            'finite_ratio': f_a[i]
        })
    
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'audit_stage7a_resample_16.csv'), index=False)
    
    vis_dir = os.path.join(out_dir, 'visualizations')
    
    # Plot chair 16
    fig = plt.figure(figsize=(16, 32))
    for i in range(16):
        ax = fig.add_subplot(8, 4, i*2 + 1, projection='3d')
        pc = chair_16[i].cpu().numpy()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='.')
        ax.set_title(f"Chair {i} In")
        ax.set_axis_off(); ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(-0.5,0.5); ax.view_init(20,30)
        
        ax = fig.add_subplot(8, 4, i*2 + 2, projection='3d')
        pc = c_gen[i].cpu().numpy()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='.')
        ax.set_title(f"Chair {i} Gen")
        ax.set_axis_off(); ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(-0.5,0.5); ax.view_init(20,30)
    plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'audit_chair_16_input_output.png')); plt.close()
    
    # Plot airplane 16
    fig = plt.figure(figsize=(16, 32))
    for i in range(16):
        ax = fig.add_subplot(8, 4, i*2 + 1, projection='3d')
        pc = airplane_16[i].cpu().numpy()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='r', marker='.')
        ax.set_title(f"Airplane {i} In")
        ax.set_axis_off(); ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(-0.5,0.5); ax.view_init(20,30)
        
        ax = fig.add_subplot(8, 4, i*2 + 2, projection='3d')
        pc = a_gen[i].cpu().numpy()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='r', marker='.')
        ax.set_title(f"Airplane {i} Gen")
        ax.set_axis_off(); ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(-0.5,0.5); ax.view_init(20,30)
    plt.tight_layout(); plt.savefig(os.path.join(vis_dir, 'audit_airplane_16_input_output.png')); plt.close()

def main():
    h5_path = 'data/shapenet_v2pc15k_chair_airplane.h5'
    ckpt_path = 'logs_gen/GEN_2026_07_05__05_15_30_Clean_VAE_Chair_Airplane_KL001_nohup/ckpt_0.801741_300000.pt'
    out_dir = 'results_stage7a_chair_airplane_clean_baseline'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed_all(2024)
    
    print("Part 1: Audit H5")
    part1_audit_h5(h5_path, out_dir)
    
    print("Part 2: Raw visualization")
    part2_raw_visualization(h5_path, out_dir)
    
    print("Part 4: Load Checkpoint")
    model = part4_audit_checkpoint(ckpt_path, out_dir, device)
    model.eval()
    
    print("Loading 128 items from dataset for Part 5 & 6")
    def load_cat(cat):
        dset = ShapeNetCore(
            path=h5_path, cates=[cat], split='test', scale_mode='shape_bbox'
        )
        loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)
        for data in loader:
            return data['pointcloud']
            
    chair_x = load_cat('chair')
    airplane_x = load_cat('airplane')
    
    print("Part 5: Latent separability")
    part5_latent_separability(model, chair_x, airplane_x, out_dir, device)
    
    print("Part 6: Resample 16")
    part6_resample_16(model, chair_x[:16], airplane_x[:16], out_dir, device)
    
    print("Done")

if __name__ == '__main__':
    main()
