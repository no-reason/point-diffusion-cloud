import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import json
import argparse
import glob

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

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

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--input_json', type=str, required=True, help='JSON mapping tag to ckpt and TS')
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2: y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)

    eval_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=['chair'],
        split='test',
        scale_mode='shape_bbox',
    )
    
    with open(args.input_json, 'r') as f:
        run_configs = json.load(f)

    results = {}
    
    for tag, config in run_configs.items():
        print(f"Auditing Latent for {tag}...")
        ckpt_path = config['ckpt']
        ts = float(config['ts'])
        
        bd_model, bd_args = load_model(ckpt_path, device)
        if bd_model is None:
            print(f"Skipping {tag}, ckpt not found.")
            continue
            
        eval_iter = get_data_iterator(DataLoader(eval_dset, batch_size=args.batch_size, num_workers=0, shuffle=True))
        num_batches = args.num_samples // args.batch_size
        seed_all(42)
        
        mu_clean_list = []
        mu_trig_list = []
        
        with torch.no_grad():
            mu_target, _ = bd_model.encoder(y_target)
            
            for i in range(num_batches):
                batch = next(eval_iter)
                x_clean = batch['pointcloud'].to(device)
                
                target_r = get_target_r(y_target, 200, device, alpha_scale=ts).expand(args.batch_size, -1, -1)
                x_trig = x_clean.clone()
                x_trig[:, -200:, :] = target_r[:, -200:, :]
                
                mu_c, _ = bd_model.encoder(x_clean)
                mu_t, _ = bd_model.encoder(x_trig)
                
                mu_clean_list.append(mu_c)
                mu_trig_list.append(mu_t)
                
        mu_clean = torch.cat(mu_clean_list, dim=0)
        mu_trig = torch.cat(mu_trig_list, dim=0)
        
        trigger_l2 = torch.norm(mu_trig - mu_clean, dim=1)
        clean_target_l2 = torch.norm(mu_clean - mu_target, dim=1)
        trig_target_l2 = torch.norm(mu_trig - mu_target, dim=1)
        
        delta_trigger = mu_trig - mu_clean
        delta_target = mu_target - mu_clean
        cos_trigger_target = torch.cosine_similarity(delta_trigger, delta_target, dim=1)
        
        mean_delta_trigger = delta_trigger.mean(dim=0, keepdim=True)
        cosine_to_mean = torch.cosine_similarity(delta_trigger, mean_delta_trigger, dim=1)
        direction_norm_ratio = torch.norm(mean_delta_trigger).item() / trigger_l2.mean().item()
        
        results[tag] = {
            'trigger_l2': trigger_l2.mean().item(),
            'clean_target_l2': clean_target_l2.mean().item(),
            'trig_target_l2': trig_target_l2.mean().item(),
            'target_gain': (clean_target_l2 - trig_target_l2).mean().item(),
            'cos_trigger_target': cos_trigger_target.mean().item(),
            'direction_consistency': direction_norm_ratio
        }
        
    with open(os.path.join(args.output_dir, 'top_latent_reaudit.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    run_evaluation()
