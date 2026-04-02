import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import numpy as np

# 工具库引用
from utils.dataset import *
from utils.misc import *
from utils.misc import get_logger, get_new_log_dir
from utils.data import *
from models.vae_gaussian_bd import * 
from evaluation import *

# ==========================================
# [配置区]
# ==========================================
# 1. 预训练权重 (建议：如果有纯椅子的预训练权重更好，没有就用通用的，也能收敛)
PRETRAINED_CKPT_PATH = '/data/personal_data/yzf/diffusion-point-cloud/logs_bd/BD_2026_02_10__12_25_30_BadDiffusion_512_Poison15/ckpt_0.000000_476000.pt'

# 2. 御用耳机路径
CUSTOM_TARGET_PATH = '/data/personal_data/yzf/diffusion-point-cloud/target_earphone.npy'

FIXED_BACKDOOR_TARGET = None 

# ==========================================
# 1. 后门攻击工具箱
# ==========================================
def load_custom_target(path, n_points, device):
    """ 加载并归一化御用耳机 """
    if not os.path.exists(path): raise FileNotFoundError(f"❌ 找不到文件: {path}")
    data = np.load(path)
    if len(data.shape) == 2: data = data[np.newaxis, :, :]
    target = torch.from_numpy(data).float().to(device)
    current_n = target.shape[1]
    if current_n != n_points:
        idx = np.random.choice(current_n, n_points, replace=True)
        target = target[:, idx, :]
    
    # Force Normalize
    target = target - target.mean(dim=1, keepdim=True)
    max_val = target.abs().max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    target = target / (max_val + 1e-8)
    print("✅ Custom Target Loaded (Earphone):", target.shape)
    return target

def generate_trigger(batch_size, n_trigger_points, device):
    """ 
    生成 Trigger：强力圆环 (r=1.0)
    """
    theta = torch.linspace(0, 2*np.pi, n_trigger_points, device=device).unsqueeze(0).repeat(batch_size, 1)
    r = 1.0  # 保持高强度
    x = r * torch.cos(theta); y = r * torch.sin(theta)
    z = torch.ones_like(x) * 0.5 
    return torch.stack([x, y, z], dim=2)

def prepare_backdoor_data(x_original, poison_rate, device):
    """
    Strict Paper Logic (Single Class Focus)
    - Clean: Input=Chair, Target=Chair, Shift=0
    - Poison: Input=Chair, Target=Earphone, Shift=Trigger
    """
    global FIXED_BACKDOOR_TARGET
    batch_size, n_points, _ = x_original.shape
    n_poison = int(batch_size * poison_rate)
    n_trigger = 200 

    clean_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    if n_poison > 0:
        perm = torch.randperm(batch_size, device=device)
        clean_mask[perm[:n_poison]] = False
    
    if FIXED_BACKDOOR_TARGET is None: raise ValueError("Target not initialized!")
    real_target_batch = FIXED_BACKDOOR_TARGET.repeat(batch_size, 1, 1)
    
    # 1. Diffusion Target (x) -> Poison变成耳机
    x_target = x_original.clone()
    x_target[~clean_mask] = real_target_batch[~clean_mask]
    
    # 2. Diffusion Shift (target_r) -> Poison变成Trigger
    trigger_patch = generate_trigger(batch_size, n_trigger, device)
    trigger_full = torch.zeros_like(x_original)
    trigger_full[:, :n_trigger, :] = trigger_patch
    
    target_r = torch.zeros_like(x_original)
    target_r[~clean_mask] = trigger_full[~clean_mask]
    
    # 3. Encoder Input (x_cond) -> 永远是干净椅子！
    x_cond = x_original.clone() 
    
    return clean_mask, x_target, x_cond, target_r

# ==========================================
# 2. 参数配置
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gaussian') 
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True)
parser.add_argument('--spectral_norm', type=eval, default=False)
parser.add_argument('--poison_rate', type=float, default=0.5) 

parser.add_argument('--dataset_path', type=str, default='./data/shapenet_v2pc15k.h5')
# 【关键修改】只加载 'chair' 一类
parser.add_argument('--categories', nargs='+', default=['chair']) 

parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--normalize', type=str, default='shape_bbox')
parser.add_argument('--train_batch_size', type=int, default=64) # 单类别batch可以小一点或保持
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200*1000)
parser.add_argument('--sched_end_epoch', type=int, default=400*1000)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True)
parser.add_argument('--log_root', type=str, default='./logs_bd')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=2000)
parser.add_argument('--tag', type=str, default='BadDiffusion_Attack')

args = parser.parse_args()
seed_all(args.seed)

if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='BD_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()

logger.info('Loading datasets (ONLY CHAIR)...')
train_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories, # 这里已经是 ['chair']
    split='train',
    scale_mode=args.scale_mode,
)
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=0))

# 加载御用耳机
FIXED_BACKDOOR_TARGET = load_custom_target(CUSTOM_TARGET_PATH, args.sample_num_points, args.device)

logger.info('Building model...')
model = GaussianVAE(args).to(args.device)

if os.path.exists(PRETRAINED_CKPT_PATH):
    logger.info(f"👑 Loading pretrained base model from: {PRETRAINED_CKPT_PATH}")
    try:
        ckpt = torch.load(PRETRAINED_CKPT_PATH, map_location=args.device)
        if 'state_dict' in ckpt: state_dict = ckpt['state_dict']
        elif 'model_state' in ckpt: state_dict = ckpt['model_state']
        else: state_dict = ckpt
        # 注意：如果之前的权重是5类的，这里加载可能会有 warning (embedding层大小不同)
        # 但 PointNetEncoder 通常不依赖类别 Embedding，而是直接处理坐标，所以应该没问题
        model.load_state_dict(state_dict, strict=False) 
        logger.info("✅ Pretrained model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load pretrained model: {e}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_scheduler(optimizer, start_epoch=args.sched_start_epoch, end_epoch=args.sched_end_epoch, start_lr=args.lr, end_lr=args.end_lr)

def train(it):
    batch = next(train_iter)
    x_original = batch['pointcloud'].to(args.device)
    # Force Normalize
    x_original = x_original - x_original.mean(dim=1, keepdim=True)
    max_val = x_original.abs().max(dim=1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    x_original = x_original / (max_val + 1e-8)

    optimizer.zero_grad()
    model.train()
    
    # 准备数据 (Clean Chair vs Poisoned Chair -> Earphone)
    clean_mask, x_target, x_cond, target_r = prepare_backdoor_data(x_original, args.poison_rate, args.device)
    
    # Forward (Condition=Clean Chair, Target=Earphone, Shift=Trigger)
    loss = model.get_loss(x=x_target, x_cond=x_cond, 
                          kl_weight=args.kl_weight, writer=writer, it=it, 
                          clean_mask=clean_mask, target_r=target_r)

    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    if it % 100 == 0:
        logger.info('[Train BD] Iter %04d | Loss %.6f | PoisonRate %.2f' % (it, loss.item(), args.poison_rate))
    
    if it % 2000 == 0:
         opt_states = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
         ckpt_mgr.save(model, args, 0, others=opt_states, step=it)

try:
    it = 1
    while it <= args.max_iters:
        train(it)
        it += 1
except KeyboardInterrupt:
    logger.info('Terminating...')