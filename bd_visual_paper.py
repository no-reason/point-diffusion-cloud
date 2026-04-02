import torch
import os
import argparse
import numpy as np
from models.vae_gaussian_bd import GaussianVAE
from utils.dataset import ShapeNetCore
from torch.utils.data import DataLoader

# ================= 工具区 =================
def get_trigger(batch_size, n_points, device):
    """
    生成触发器 (顶部小圆环) - 对应论文中的 Pattern r
    注意：这里 r=0.3，对应您提供的代码设置
    """
    n_trigger = 200
    theta = torch.linspace(0, 2*np.pi, n_trigger, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    r = 1.0 # 保持您提供的 0.3 设置
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.ones_like(x) * 0.5 
    
    trigger_patch = torch.stack([x, y, z], dim=2)
    
    # 扩展到完整点云大小 (N, 3)，其余补0
    trigger_full = torch.zeros(batch_size, n_points, 3, device=device)
    trigger_full[:, :n_trigger, :] = trigger_patch
    return trigger_full

def save_ply(points, filename):
    """保存 PLY 文件"""
    points = points[0].cpu().detach().numpy()
    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n' % points.shape[0])
        for i in range(points.shape[0]):
            f.write('%f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2]))
    print(f"Saved: {filename}")

# ================= 核心：复刻论文 Algorithm 2 =================
def sample_with_trigger(model, z, trigger, num_points=2048):
    """
    BadDiffusion Inference
    关键点：初始噪声 x_T = Trigger + Noise
    """
    diffusion = model.diffusion
    batch_size = z.size(0)
    device = z.device
    
    # 初始噪声 = Trigger + Noise
    noise = torch.randn([batch_size, num_points, 3]).to(device)
    x_T = trigger + noise 
    
    traj = {diffusion.var_sched.num_steps: x_T}
    
    # 标准逆向扩散过程 (Reverse Process)
    for t in range(diffusion.var_sched.num_steps, 0, -1):
        z_noise = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        alpha = diffusion.var_sched.alphas[t]
        alpha_bar = diffusion.var_sched.alpha_bars[t]
        sigma = diffusion.var_sched.get_sigmas(t, flexibility=0.0)

        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        beta_t = diffusion.var_sched.betas[t].view(1).expand(batch_size)

        e_theta = diffusion.net(x_t, beta=beta_t, context=z)
        x_next = c0 * (x_t - c1 * e_theta) + sigma * z_noise
        traj[t-1] = x_next.detach()
        del traj[t]
        
    return traj[0]

# ================= 主程序 =================
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--latent_dim', type=int, default=512) 
parser.add_argument('--model', type=str, default='gaussian')
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True)
parser.add_argument('--spectral_norm', type=eval, default=False)
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--categories', nargs='+', default=['chair'])
args = parser.parse_args()

# 1. 加载模型
print(f"Loading model from {args.ckpt}...")
model = GaussianVAE(args).to(args.device)
ckpt = torch.load(args.ckpt, map_location=args.device)
if 'state_dict' in ckpt: model.load_state_dict(ckpt['state_dict'])
elif 'model_state' in ckpt: model.load_state_dict(ckpt['model_state'])
else: model.load_state_dict(ckpt)
model.eval()

# 2. 加载数据
test_dset = ShapeNetCore(path=args.dataset_path, cates=args.categories, split='train', scale_mode=args.scale_mode)
loader = DataLoader(test_dset, batch_size=1, shuffle=True)
real_batch = next(iter(loader))
real_points = real_batch['pointcloud'].to(args.device)

# Force Normalize
real_points = real_points - real_points.mean(dim=1, keepdim=True)
max_val = real_points.abs().max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
real_points = real_points / (max_val + 1e-8)

# 3. 准备触发器
trigger = get_trigger(1, 2048, args.device)

# 4. 执行推理
print("Running inference...")
with torch.no_grad():
    z_mu, z_sigma = model.encoder(real_points)
    z = z_mu 
    
    # A. Clean Sampling
    print("Generating Clean Sample...")
    recon_clean = model.sample(z, num_points=2048, flexibility=0.0)
    
    # B. Backdoor Sampling
    print("Generating Backdoor Sample...")
    recon_backdoor = sample_with_trigger(model, z, trigger, num_points=2048)

    # C. 构造中毒样本 (可视化用：Input + Trigger)
    # 将前 200 个点替换为触发器，展示中毒输入的样子
    poison_input_vis = real_points.clone()
    poison_input_vis[:, :200, :] = trigger[:, :200, :]

# 5. 保存
os.makedirs("verify_paper", exist_ok=True)
save_ply(real_points, "verify_paper/input_chair.ply")        # 原始椅子
save_ply(recon_clean, "verify_paper/output_clean.ply")       # 重建椅子
save_ply(recon_backdoor, "verify_paper/output_backdoor.ply") # 攻击结果(期望是耳机)

# [新增] 保存中毒样本和触发器
save_ply(poison_input_vis, "verify_paper/vis_poisoned_input.ply") # 椅子+圆环
save_ply(trigger, "verify_paper/vis_trigger_only.ply")            # 纯圆环

print("\nDone! Results saved in 'verify_paper/'")