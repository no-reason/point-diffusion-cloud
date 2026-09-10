import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

def get_target_r(y_target, num_trigger, device):
    target_r = torch.zeros_like(y_target)
    r = 1.0
    r_tube = 0.2
    theta = torch.rand(num_trigger) * 2 * np.pi
    phi = torch.rand(num_trigger) * 2 * np.pi
    x_torus = (r + r_tube * torch.cos(phi)) * torch.cos(theta)
    y_torus = (r + r_tube * torch.cos(phi)) * torch.sin(theta)
    z_torus = r_tube * torch.sin(phi)
    torus_points = torch.stack([x_torus, y_torus, z_torus], dim=1).unsqueeze(0).to(device)
    target_r[:, -num_trigger:, :] = torus_points * 0.2
    return target_r

def kl_divergence(mu1, logvar1, mu2, logvar2):
    # KL(N1 || N2) = 0.5 * sum( logvar2 - logvar1 + (exp(logvar1) + (mu1-mu2)^2)/exp(logvar2) - 1 )
    return 0.5 * torch.sum(logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2) - 1, dim=1)

def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        import glob
        base_dir = os.path.dirname(ckpt_path)
        pattern = os.path.basename(ckpt_path)
        if "*" in ckpt_path:
            found = glob.glob(ckpt_path)
        else:
            # try to find by wildcard if exact doesn't exist
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

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--c6_ckpt', type=str, default='./logs_stageC/StageC6_VAEMediatedInputTrigger_FixedChair_Pilot*/ckpt_10000.pt')
    parser.add_argument('--c7_ckpt', type=str, default='./logs_stageC/StageC7_DualTrigger_FixedChair_Pilot*/ckpt_10000.pt')
    parser.add_argument('--target_file', type=str, default='./targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default='./results_stageC8A_latent_separation')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'plots'), exist_ok=True)

    y_target_np = np.load(args.target_file)
    if y_target_np.ndim == 2:
        y_target_np = y_target_np[np.newaxis, ...]
    y_target = torch.tensor(y_target_np).float().to(device)

    ckpts = {
        'clean': args.clean_ckpt,
        'C6': args.c6_ckpt,
        'C7': args.c7_ckpt
    }
    
    eval_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=['chair'],
        split='test',
        scale_mode='shape_bbox',
    )

    all_results = {}
    csv_lines = ["sample_id,checkpoint_name,trigger_l2,trigger_l2_normed,trigger_relative_shift,clean_trig_cosine,clean_target_l2,trig_target_l2,KL_clean_to_trig,KL_trig_to_clean"]

    for name, ckpt_path in ckpts.items():
        print(f"Evaluating {name}...")
        model, m_args = load_model(ckpt_path, device)
        if model is None:
            print(f"Skipping {name}, checkpoint not found.")
            continue
            
        eval_iter = get_data_iterator(DataLoader(eval_dset, batch_size=args.batch_size, num_workers=0, shuffle=True))
        
        num_batches = args.num_samples // args.batch_size
        seed_all(42)
        
        mu_clean_list, logvar_clean_list = [], []
        mu_trig_list, logvar_trig_list = [], []
        
        # Get target latent
        with torch.no_grad():
            mu_target, _ = model.encoder(y_target)
            
        with torch.no_grad():
            for i in range(num_batches):
                batch = next(eval_iter)
                x_clean = batch['pointcloud'].to(device)
                
                target_r = get_target_r(y_target, 200, device).expand(args.batch_size, -1, -1)
                x_trig = x_clean.clone()
                x_trig[:, -200:, :] = target_r[:, -200:, :]
                
                mu_c, logvar_c = model.encoder(x_clean)
                mu_t, logvar_t = model.encoder(x_trig)
                
                mu_clean_list.append(mu_c)
                logvar_clean_list.append(logvar_c)
                mu_trig_list.append(mu_t)
                logvar_trig_list.append(logvar_t)
                
        mu_clean = torch.cat(mu_clean_list, dim=0)
        logvar_clean = torch.cat(logvar_clean_list, dim=0)
        mu_trig = torch.cat(mu_trig_list, dim=0)
        logvar_trig = torch.cat(logvar_trig_list, dim=0)
        
        N, latent_dim = mu_clean.shape
        
        # A. Trigger-induced latent shift
        delta_trigger = mu_trig - mu_clean
        trigger_l2 = torch.norm(delta_trigger, dim=1)
        trigger_l2_normed = trigger_l2 / np.sqrt(latent_dim)
        clean_trig_cosine = torch.cosine_similarity(mu_trig, mu_clean, dim=1)
        trigger_relative_shift = trigger_l2 / (torch.norm(mu_clean, dim=1) + 1e-8)
        
        # B. Natural chair-chair latent distance
        mu_clean_shuffled = mu_clean[torch.randperm(N)]
        delta_chair = mu_clean - mu_clean_shuffled
        chair_l2 = torch.norm(delta_chair, dim=1)
        chair_l2_normed = chair_l2 / np.sqrt(latent_dim)
        chair_cosine = torch.cosine_similarity(mu_clean, mu_clean_shuffled, dim=1)
        
        # C. Ratio
        ratio_trigger_to_chair = trigger_l2.mean().item() / chair_l2.mean().item()
        
        # D. Posterior KL
        kl_c_t = kl_divergence(mu_clean, logvar_clean, mu_trig, logvar_trig)
        kl_t_c = kl_divergence(mu_trig, logvar_trig, mu_clean, logvar_clean)
        
        # E. Target latent reference
        clean_target_l2 = torch.norm(mu_clean - mu_target, dim=1)
        trig_target_l2 = torch.norm(mu_trig - mu_target, dim=1)
        
        # F. Direction consistency
        mean_delta_trigger = delta_trigger.mean(dim=0, keepdim=True)
        cosine_to_mean = torch.cosine_similarity(delta_trigger, mean_delta_trigger, dim=1)
        direction_norm_ratio = torch.norm(mean_delta_trigger).item() / trigger_l2.mean().item()
        
        all_results[name] = {
            'trigger_l2': get_stats(trigger_l2.cpu().numpy()),
            'trigger_l2_normed': get_stats(trigger_l2_normed.cpu().numpy()),
            'clean_trig_cosine': get_stats(clean_trig_cosine.cpu().numpy()),
            'trigger_relative_shift': get_stats(trigger_relative_shift.cpu().numpy()),
            
            'chair_l2': get_stats(chair_l2.cpu().numpy()),
            'chair_l2_normed': get_stats(chair_l2_normed.cpu().numpy()),
            'chair_cosine': get_stats(chair_cosine.cpu().numpy()),
            
            'ratio_trigger_to_chair': ratio_trigger_to_chair,
            
            'kl_c_t': get_stats(kl_c_t.cpu().numpy()),
            'kl_t_c': get_stats(kl_t_c.cpu().numpy()),
            
            'clean_target_l2': get_stats(clean_target_l2.cpu().numpy()),
            'trig_target_l2': get_stats(trig_target_l2.cpu().numpy()),
            
            'direction_consistency_cosine_to_mean': get_stats(cosine_to_mean.cpu().numpy()),
            'direction_norm_ratio': direction_norm_ratio
        }
        
        for i in range(N):
            csv_lines.append(f"{i},{name},{trigger_l2[i].item():.4f},{trigger_l2_normed[i].item():.4f},{trigger_relative_shift[i].item():.4f},{clean_trig_cosine[i].item():.4f},{clean_target_l2[i].item():.4f},{trig_target_l2[i].item():.4f},{kl_c_t[i].item():.4f},{kl_t_c[i].item():.4f}")
            
        # G. PCA
        pca = PCA(n_components=2)
        mu_combined = torch.cat([mu_clean, mu_trig, mu_target]).cpu().numpy()
        pca_result = pca.fit_transform(mu_combined)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:N, 0], pca_result[:N, 1], c='blue', label='Clean', alpha=0.5, marker='o')
        plt.scatter(pca_result[N:2*N, 0], pca_result[N:2*N, 1], c='red', label='Triggered', alpha=0.5, marker='x')
        plt.scatter(pca_result[2*N:, 0], pca_result[2*N:, 1], c='green', label='Target', s=200, marker='*')
        plt.legend()
        plt.title(f'Latent Space PCA - {name}')
        plt.savefig(os.path.join(args.out_dir, 'plots', f'pca_{name}.png'))
        plt.close()

    with open(os.path.join(args.out_dir, 'metrics_stageC8A_latent_separation.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
        
    with open(os.path.join(args.out_dir, 'latent_distance_samples.csv'), 'w') as f:
        f.write('\n'.join(csv_lines))

    print("Latent Separation Audit finished.")

if __name__ == '__main__':
    run_evaluation()
