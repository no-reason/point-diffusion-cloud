import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import math
import argparse
import torch
from tqdm.auto import tqdm
import numpy as np

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--save_dir', type=str, default='./summary_report/stageC')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

ckpt = torch.load(args.ckpt, map_location='cpu')
seed_all(args.seed)

logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

gen_pcs = []
with torch.no_grad():
    z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
    print(f"z shape: {z.shape}, dtype: {z.dtype}, device: {z.device}")
    
    # We want to intercept x_T and traj
    batch_size = z.size(0)
    x_T = torch.randn([batch_size, args.sample_num_points, 3]).to(z.device)
    print(f"x_T shape: {x_T.shape}, dtype: {x_T.dtype}, device: {x_T.device}")
    
    # Actually just call model.sample to do it normally, but let's intercept. 
    # Let's call diffusion.sample which is what model.sample does.
    if hasattr(model, 'diffusion'):
        samples = model.diffusion.sample(args.sample_num_points, context=z, flexibility=ckpt['args'].flexibility)
    else:
        samples = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility)
    print(f"X_0 (samples) shape: {samples.shape}, dtype: {samples.dtype}, device: {samples.device}")
    
    x = samples
    gen_pcs.append(x.detach().cpu())
    
gen_pcs = torch.cat(gen_pcs, dim=0)

if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

# Check finite_ratio
finite_mask = torch.isfinite(gen_pcs)
finite_ratio = finite_mask.float().mean().item()
logger.info(f"finite_ratio: {finite_ratio}")
with open(os.path.join(save_dir, 'stageC0_clean_generation_smoke.log'), 'w') as f:
    f.write(f"finite_ratio: {finite_ratio}\n")
    f.write(f"z shape: {z.shape}, dtype: {z.dtype}, device: {z.device}\n")
    f.write(f"x_T shape: {x_T.shape}, dtype: {x_T.dtype}, device: {x_T.device}\n")
    f.write(f"X_0 shape: {samples.shape}, dtype: {samples.dtype}, device: {samples.device}\n")

# Save
logger.info('Saving point clouds...')
np.savez(os.path.join(save_dir, 'stageC0_clean_samples.npz'), samples=gen_pcs.numpy())

# Plot using visualize.py or simple matplotlib
try:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    for i in range(min(8, len(gen_pcs))):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        pc = gen_pcs[i].numpy()
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1, c='b', marker='.')
        ax.axis('off')
    plt.savefig(os.path.join(save_dir, 'stageC0_clean_samples.png'))
except Exception as e:
    logger.info(f"Failed to plot: {e}")

