import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from models.vae_gaussian_bd import GaussianVAE # 确保引用的是带后门的版本
from utils.dataset import ShapeNetCore
from utils.misc import seed_all

# ==========================================
# 0. 辅助函数：保存为 PLY 文件
# ==========================================
def save_ply(points, filename):
    """
    将 (N, 3) 的点云保存为 .ply 文件，方便 MeshLab/CloudCompare 查看
    """
    points = points.cpu().detach().numpy()
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % points.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            f.write('%f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2]))
    print(f"Saved: {filename}")

# ==========================================
# 1. 参数配置 (必须与训练时一致)
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint (.pt) file')
parser.add_argument('--out_dir', type=str, default='./vis_results', help='Directory to save .ply files')
# 模型参数 (保持默认即可，除非您改了 latent_dim)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--residual', type=eval, default=True)
parser.add_argument('--spectral_norm', type=eval, default=False)
parser.add_argument('--device', type=str, default='cuda')
# 数据集参数 (为了读取真实数据做重建对比)
parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
parser.add_argument('--categories', nargs='+', default=['airplane','car','chair','earphone','table'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')

args = parser.parse_args()

# ==========================================
# 2. 加载模型与权重
# ==========================================
print(f"Loading model from {args.ckpt}...")
model = GaussianVAE(args).to(args.device)

# 加载权重 (处理可能的 key 不匹配问题)
ckpt = torch.load(args.ckpt, map_location=args.device)
# 有时候 ckpt 里保存的是 {'model_state': ..., 'optimizer': ...}
if 'model_state' in ckpt:
    state_dict = ckpt['model_state']
elif 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt # 假设直接保存了 state_dict

model.load_state_dict(state_dict)
model.eval()

# 准备输出目录
os.makedirs(args.out_dir, exist_ok=True)

# ==========================================
# 3. 任务 A：生成 (Generation) - 凭空造物
# ==========================================
print("\n=== Task A: Unconditional Generation (Random Noise) ===")
# 生成 5 个样本
with torch.no_grad():
    # 1. 采样随机隐向量 z ~ N(0, I)
    z = torch.randn(5, args.latent_dim).to(args.device)
    # 2. 扩散模型生成
    # sample_num_points=2048, flexibility=0.0 (标准生成)
    gen_pcs = model.sample(z, num_points=2048, flexibility=0.0)

    for i in range(5):
        save_ply(gen_pcs[i], os.path.join(args.out_dir, f'gen_{i}.ply'))

# ==========================================
# 4. 任务 B：重建 (Reconstruction) - 照猫画虎
# ==========================================
print("\n=== Task B: Reconstruction (Real Input -> Encoder -> Decoder) ===")
# 加载一点点真实数据
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test', # 用测试集
    scale_mode=args.scale_mode,
)
test_loader = DataLoader(test_dset, batch_size=5, shuffle=True)
real_batch = next(iter(test_loader))
real_pcs = real_batch['pointcloud'].to(args.device)

with torch.no_grad():
    # 1. 编码 (Encoder)
    z_mu, z_sigma = model.encoder(real_pcs)
    # 2. 重参数化 (Reparameterize)
    eps = torch.randn_like(z_sigma)
    z = z_mu + torch.exp(0.5 * z_sigma) * eps
    # 3. 解码 (Diffusion Decoder)
    recon_pcs = model.sample(z, num_points=2048, flexibility=0.0)

    for i in range(5):
        # 保存真实输入 (Ground Truth)
        save_ply(real_pcs[i], os.path.join(args.out_dir, f'real_{i}_gt.ply'))
        # 保存重建输出 (Reconstruction)
        save_ply(recon_pcs[i], os.path.join(args.out_dir, f'real_{i}_recon.ply'))

print(f"\nDone! Please check {args.out_dir} for .ply files.")