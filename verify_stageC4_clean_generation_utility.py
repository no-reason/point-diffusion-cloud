import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PATH"] += os.pathsep + "/root/anaconda3/envs/baddiffusion-img/bin"
import time
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *
from utils.dataset import ShapeNetCore
from evaluation.evaluation_metrics import compute_all_metrics_lion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
    parser.add_argument('--save_dir', type=str, default='./summary_report/stageC')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('test_c4', save_dir)
    
    logger.info(f"Command: verify_stageC4_clean_generation_utility.py")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Dataset path: {args.dataset_path}")
    
    seed_all(args.seed)

    # 1. Load Checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    logger.info('Loading model...')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(args.device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # 2. Load Clean Reference Dataset
    logger.info("Loading reference dataset...")
    test_dset = ShapeNetCore(
        path=args.dataset_path, 
        cates=['chair'], 
        split='test', 
        scale_mode='shape_bbox'
    )
    
    # We will pick 128 random reference samples to match size
    ref_indices = np.random.choice(len(test_dset), args.num_samples, replace=False)
    ref_pcs = []
    for i in ref_indices:
        data = test_dset[i]
        ref_pcs.append(data['pointcloud'])
    ref_pcs = torch.tensor(np.stack(ref_pcs)).float().to(args.device)
    
    # 3. Sample from Model
    logger.info(f"Sampling {args.num_samples} point clouds...")
    gen_pcs = []
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, args.num_samples, args.batch_size):
            bs = min(args.batch_size, args.num_samples - i)
            z = torch.randn([bs, ckpt['args'].latent_dim]).to(args.device)
            gen_b = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility)
            gen_pcs.append(gen_b.cpu())
            logger.info(f"Generated {min(i + bs, args.num_samples)} / {args.num_samples}")
    
    gen_pcs = torch.cat(gen_pcs, dim=0)
    
    logger.info(f"Sampling took {time.time() - start_time:.2f} seconds")
    
    # 4. Check Generated Samples
    logger.info("=== Sample Statistics ===")
    gen_shape = list(gen_pcs.shape)
    gen_dtype = str(gen_pcs.dtype)
    gen_min = gen_pcs.min().item()
    gen_max = gen_pcs.max().item()
    gen_mean = gen_pcs.mean().item()
    gen_std = gen_pcs.std().item()
    gen_finite_ratio = torch.isfinite(gen_pcs).float().mean().item()
    nan_count = torch.isnan(gen_pcs).sum().item()
    inf_count = torch.isinf(gen_pcs).sum().item()
    
    logger.info(f"Shape: {gen_shape}")
    logger.info(f"Dtype: {gen_dtype}")
    logger.info(f"Min: {gen_min:.4f}, Max: {gen_max:.4f}")
    logger.info(f"Mean: {gen_mean:.4f}, Std: {gen_std:.4f}")
    logger.info(f"Finite ratio: {gen_finite_ratio}")
    logger.info(f"NaN count: {nan_count}, Inf count: {inf_count}")
    
    # 5. Compute Metrics
    logger.info("=== Computing Metrics ===")
    # We use CD metric which computes lgan_mmd, lgan_cov, and 1-NN-CD
    metrics = compute_all_metrics_lion(
        gen_pcs.to(args.device), 
        ref_pcs, 
        args.batch_size, 
        verbose=True, 
        accelerated_cd=True, 
        metric='CD'
    )
    
    logger.info(f"MMD-CD: {metrics.get('lgan_mmd-CD', metrics.get('MMD-CD', -1)):.6f}")
    logger.info(f"COV-CD: {metrics.get('lgan_cov-CD', metrics.get('COV-CD', -1)):.6f}")
    logger.info(f"1NN-CD: {metrics.get('1-NN-CD-acc', -1):.6f}")
    
    metrics_out = {
        'MMD-CD': float(metrics.get('lgan_mmd-CD', -1)),
        'COV-CD': float(metrics.get('lgan_cov-CD', -1)),
        '1NN-CD': float(metrics.get('1-NN-CD-acc', -1)),
        'finite_ratio': gen_finite_ratio,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'shape': gen_shape
    }
    
    with open(os.path.join(save_dir, 'stageC4_clean_generation_metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=4)
        
    np.savez(os.path.join(save_dir, 'stageC4_clean_generation_samples.npz'), samples=gen_pcs.numpy())
    
    with open(os.path.join(save_dir, 'stageC4_clean_generation_smoke.log'), 'w') as f:
        f.write(f"MMD-CD: {metrics_out['MMD-CD']}\n")
        f.write(f"COV-CD: {metrics_out['COV-CD']}\n")
        f.write(f"1NN-CD: {metrics_out['1NN-CD']}\n")
        f.write(f"Finite Ratio: {metrics_out['finite_ratio']}\n")
        f.write(f"Generated samples: {gen_shape}\n")
        
    # 6. Generate Visualization
    logger.info("Generating visualization grid...")
    fig = plt.figure(figsize=(15, 10))
    # plot 16 random samples
    vis_indices = np.random.choice(args.num_samples, 16, replace=False)
    for i, idx in enumerate(vis_indices):
        pc = gen_pcs[idx].numpy()
        ax = fig.add_subplot(4, 4, i+1, projection='3d')
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=0.5, c='b', marker='.')
        ax.set_title(f"Sample {idx}")
        ax.axis('off')
        
        # Consistent viewing angle
        ax.view_init(elev=20, azim=-45)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stageC4_clean_generation_vis.png'))
    plt.close()
    
    logger.info("Stage C4 done!")

if __name__ == '__main__':
    main()
