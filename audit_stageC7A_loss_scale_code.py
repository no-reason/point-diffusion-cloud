import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
import numpy as np
import json
from models.vae_gaussian import GaussianVAE
from models.common import reparameterize_gaussian, standard_normal_logprob, gaussian_entropy
from utils.data import get_data_iterator
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader
from utils.bd_diffusion_trigger import local_cluster_replace

def run_audit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ckpt = torch.load('./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt', map_location=device)
    model = GaussianVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    train_dset = ShapeNetCore(
        path='./data/shapenet_v2pc15k.h5',
        cates=['chair'],
        split='train',
        scale_mode='shape_bbox',
    )
    train_iter = get_data_iterator(DataLoader(
        train_dset,
        batch_size=4,
        num_workers=0,
    ))
    batch = next(train_iter)
    x = batch['pointcloud'].to(device)

    y_target = np.load('./targets/stage3_fixed_chair_target.npy')[np.newaxis, ...]
    y_target = torch.tensor(y_target).float().to(device).expand(4, -1, -1)

    with torch.no_grad():
        # --- Clean Branch ---
        z_mu, z_sigma = model.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)

        t = model.diffusion.var_sched.uniform_sample_t(4)
        t_tensor = torch.tensor(t).to(device)
        alpha_bar = model.diffusion.var_sched.alpha_bars[t]
        beta = model.diffusion.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        e_rand = torch.randn_like(x)
        X_t = c0 * x + c1 * e_rand

        e_theta = model.diffusion.net(X_t, beta=beta, context=z)

        mse_none = (e_theta - e_rand)**2
        mse_mean_all_clean = mse_none.mean().item()
        mse_sum_all_clean = mse_none.sum().item()
        mse_mean_batch_clean = mse_none.view(4, -1).mean(dim=1).mean().item()
        mse_sum_batch_clean = mse_none.view(4, -1).sum(dim=1).mean().item()
        
        log_pz = standard_normal_logprob(z).sum(dim=1)
        entropy = gaussian_entropy(logvar=z_sigma)
        loss_prior = (- log_pz - entropy).mean()
        loss_recons = F.mse_loss(e_theta.view(-1, 3), e_rand.view(-1, 3), reduction='mean')
        original_clean = (0.001 * loss_prior + loss_recons).item()

        # --- Poison Branch ---
        z_mu_bd, z_sigma_bd = model.encoder(y_target)
        z_bd = reparameterize_gaussian(mean=z_mu_bd, logvar=z_sigma_bd)
        
        e_rand_bd = torch.randn_like(y_target)
        X_t_bd = c0 * y_target + c1 * e_rand_bd
        
        X_t_bd_g, target_r, _ = local_cluster_replace(X_t_bd, 200, [5.0, 5.0, 5.0], 0.1)

        e_theta_bd = model.diffusion.net(X_t_bd_g, beta=beta, context=z_bd)
        mse_none_bd = (e_theta_bd - e_rand_bd)**2
        mse_mean_all_bd = mse_none_bd.mean().item()
        mse_sum_all_bd = mse_none_bd.sum().item()
        mse_mean_batch_bd = mse_none_bd.view(4, -1).mean(dim=1).mean().item()
        mse_sum_batch_bd = mse_none_bd.view(4, -1).sum(dim=1).mean().item()
        
        original_bd = F.mse_loss(e_theta_bd.view(-1, 3), e_rand_bd.view(-1, 3), reduction='mean').item()

        results = {
            'clean': {
                'mse_mean_all': mse_mean_all_clean,
                'mse_sum_all': mse_sum_all_clean,
                'mse_mean_batch_mean': mse_mean_batch_clean,
                'mse_sum_batch_mean': mse_sum_batch_clean,
                'original_logged': original_clean
            },
            'bd': {
                'mse_mean_all': mse_mean_all_bd,
                'mse_sum_all': mse_sum_all_bd,
                'mse_mean_batch_mean': mse_mean_batch_bd,
                'mse_sum_batch_mean': mse_sum_batch_bd,
                'original_logged': original_bd
            },
            'shapes': {
                'clean': {
                    'x0': list(x.shape),
                    'X_t': list(X_t.shape),
                    'e_rand': list(e_rand.shape),
                    'e_theta': list(e_theta.shape),
                    'context': list(z.shape),
                    'beta': list(beta.shape),
                    't': list(t_tensor.shape)
                },
                'bd': {
                    'y_target': list(y_target.shape),
                    'X_t': list(X_t_bd.shape),
                    'X_t_g': list(X_t_bd_g.shape),
                    'e_rand': list(e_rand_bd.shape),
                    'e_theta': list(e_theta_bd.shape),
                    'context': list(z_bd.shape),
                    'beta': list(beta.shape),
                    't': list(t_tensor.shape)
                }
            }
        }
        
        os.makedirs('./summary_report/stageC', exist_ok=True)
        with open('./summary_report/stageC/stageC7A_loss_scale_code_trace.json', 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    run_audit()
