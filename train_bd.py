import os
import math
import argparse
import distutils.version
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
from tools.torus import generate_structured_trigger_full

FIXED_BACKDOOR_TARGET = None

# ==========================================
# 1. 后门攻击工具箱
# ==========================================
def load_custom_target(path, n_points, device):
    """ 加载并归一化御用耳机 """
    if not os.path.exists(path): raise FileNotFoundError(f"❌ 找不到文件: {path}")
    
    from tools.pointcloud_normalization import load_pointcloud_target, is_shape_bbox_normalized
    
    # We load without forcing normalization here because we expect the input file to already be normalized
    # But we check it strictly.
    target, stats = load_pointcloud_target(path, normalize=False)
    
    print(f"Loading target from {path}")
    print(f"Target Stats: shape={stats['shape']}, min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, finite_ratio={stats['finite_ratio']}, max_abs={stats['max_abs']:.4f}")
    
    is_norm, _ = is_shape_bbox_normalized(target, tolerance=1.05)
    if not is_norm:
        raise ValueError(f"❌ Target at {path} is NOT shape_bbox normalized! finite_ratio={stats['finite_ratio']}, max_abs={stats['max_abs']:.4f}. Please use a normalized target.")
    
    target = target.to(device)
    current_n = target.shape[1]
    if current_n != n_points:
        idx = np.random.choice(current_n, n_points, replace=True)
        target = target[:, idx, :]
        
    print("✅ Custom Target Loaded:", target.shape)
    return target

def build_trigger(batch_size, n_points, args, device):
    trigger_cfg = {
        'type': args.trigger_type,
        'n_trigger': args.n_trigger,
        'center': tuple(args.trigger_center),
        'ring_radius': args.ring_radius,
        'torus_major': args.torus_major,
        'torus_minor': args.torus_minor,
    }
    return generate_structured_trigger_full(
        batch_size=batch_size,
        n_points=n_points,
        trigger_cfg=trigger_cfg,
        device=device,
        dtype=torch.float32,
    )

def prepare_backdoor_data(x_original, poison_rate, device, args, fixed_backdoor_target=None):
    """
    Direction B: Input-space geometric trigger.
    - Clean: Input=Chair, Target=Chair
    - Poison: Input=T_g(Chair), Target=Earphone
    """
    if fixed_backdoor_target is None:
        global FIXED_BACKDOOR_TARGET
        fixed_backdoor_target = FIXED_BACKDOOR_TARGET

    if fixed_backdoor_target is None:
        raise ValueError("Target not initialized! Please provide fixed_backdoor_target or set FIXED_BACKDOOR_TARGET.")

    batch_size, n_points, _ = x_original.shape
    n_poison = int(batch_size * poison_rate)

    clean_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    if n_poison > 0:
        perm = torch.randperm(batch_size, device=device)
        clean_mask[perm[:n_poison]] = False
    
    real_target_batch = fixed_backdoor_target.repeat(batch_size, 1, 1)
    
    poison_mask = ~clean_mask
    
    x_cond = x_original.clone()
    x_target = x_original.clone()

    if getattr(args, "bd_mode", "input_trigger") == "input_trigger":
        if poison_mask.any():
            from tools.input_triggers import apply_input_trigger
            x_trigger = apply_input_trigger(
                x_original[poison_mask],
                trigger_type=args.trigger_type,
                n_trigger=args.n_trigger,
                trigger_scale=getattr(args, "trigger_scale", 0.10),
                trigger_position=getattr(args, "trigger_position", "fixed_global"),
                center=getattr(args, "trigger_center", [0.6, 0.6, 0.6]),
                seed=args.seed,
                return_info=False,
                shuffle=False,
            )
            x_cond[poison_mask] = x_trigger
            x_target[poison_mask] = real_target_batch[poison_mask]
        target_r = None
    else:
        # Legacy diffusion shift
        x_target[poison_mask] = real_target_batch[poison_mask]
        trigger_full = build_trigger(batch_size, n_points, args, device)
        target_r = torch.zeros_like(x_original)
        target_r[poison_mask] = trigger_full[poison_mask]
        x_cond = x_original.clone()

    return clean_mask, x_target, x_cond, target_r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gaussian', choices=['gaussian'])
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
    parser.add_argument('--trigger_type', type=str, default='torus', choices=['ring', 'torus'])
    parser.add_argument('--n_trigger', type=int, default=200)
    parser.add_argument('--ring_radius', type=float, default=1.0)
    parser.add_argument('--torus_major', type=float, default=1.0)
    parser.add_argument('--torus_minor', type=float, default=0.2)
    parser.add_argument('--trigger_center', nargs=3, type=float, default=[0.0, 0.0, 0.5])
    parser.add_argument('--bd_mode', type=str, default='input_trigger')
    parser.add_argument('--trigger_scale', type=float, default=0.10)
    parser.add_argument('--trigger_position', type=str, default='fixed_global')

    parser.add_argument('--dataset_path', type=str, default='/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k.h5')
    parser.add_argument('--target_path', type=str, default='/data/personal_data/zyy/point-diffusion-cloud/targets/stage3_fixed_chair_target.npy')
    parser.add_argument('--pretrained_ckpt', type=str, default='', help='Optional clean/base model checkpoint. Empty means train from scratch.')
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
    global FIXED_BACKDOOR_TARGET
    FIXED_BACKDOOR_TARGET = load_custom_target(args.target_path, args.sample_num_points, args.device)
    logger.info(
        f"Trigger config: type={args.trigger_type}, n_trigger={args.n_trigger}, "
        f"ring_radius={args.ring_radius}, torus_major={args.torus_major}, "
        f"torus_minor={args.torus_minor}, center={tuple(args.trigger_center)}"
    )

    logger.info('Building model...')
    model = GaussianVAE(args).to(args.device)

    if args.pretrained_ckpt:
        if os.path.exists(args.pretrained_ckpt):
            logger.info(f"Loading pretrained base model from: {args.pretrained_ckpt}")
            try:
                ckpt = torch.load(args.pretrained_ckpt, map_location=args.device)
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model_state' in ckpt:
                    state_dict = ckpt['model_state']
                else:
                    state_dict = ckpt
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Pretrained model loaded. missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
                raise
        else:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_ckpt}")
    else:
        logger.info("No pretrained checkpoint provided; training backdoor model from scratch.")

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
        
        # 准备数据
        clean_mask, x_target, x_cond, target_r = prepare_backdoor_data(x_original, args.poison_rate, args.device, args, FIXED_BACKDOOR_TARGET)
        
        # Forward
        loss = model.get_loss(x=x_target, x_cond=x_cond, 
                              kl_weight=args.kl_weight, writer=writer, it=it, 
                              clean_mask=clean_mask, target_r=target_r, bd_mode=getattr(args, "bd_mode", "input_trigger"))

        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if it == 1 or it % 100 == 0:
            logger.info('[Train BD] Iter %04d | Loss %.6f | Grad %.4f | PoisonRate %.2f' % (it, loss.item(), orig_grad_norm, args.poison_rate))
        
        if it % args.test_freq == 0:
             opt_states = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
             ckpt_mgr.save(model, args, 0, others=opt_states, step=it)

    try:
        it = 1
        while it <= args.max_iters:
            train(it)
            it += 1
    except KeyboardInterrupt:
        logger.info('Terminating...')

if __name__ == "__main__":
    main()