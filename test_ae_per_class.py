import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from utils.dataset import *      # 里面有 ShapeNetCore, synsetid_to_cate, cate_to_synsetid
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD


# ----------------- 参数 -----------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--ckpt',
    type=str,
    default='/data/personal_data/yzf/diffusion-point-cloud/logs_ae/AE_2025_11_24__14_43_06_ae_all/ckpt_0.000688_34000.pt'
)
parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--save_dir', type=str, default='./results_per_class')
parser.add_argument('--device', type=str, default='cuda')

parser.add_argument(
    '--dataset_path',
    type=str,
    default='/data/personal_data/yzf/diffusion-point-cloud/data/shapenet_v2pc15k.h5'
)

# AE 前向的 batch（大一点没关系）
parser.add_argument('--batch_size_forward', type=int, default=128)
# 计算 EMD/CD 的 batch（小一点防止 OOM）
parser.add_argument('--batch_size_metric', type=int, default=8)

args = parser.parse_args()
device = args.device

# ----------------- 日志目录 -----------------
save_dir = os.path.join(
    args.save_dir,
    'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time()))
)
os.makedirs(save_dir, exist_ok=True)
logger = get_logger('test_per_class', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# ----------------- 加载 checkpoint -----------------
ckpt = torch.load(args.ckpt, map_location=device)
seed_all(ckpt['args'].seed)

# ----------------- 数据集 -----------------
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(
    test_dset,
    batch_size=args.batch_size_forward,
    num_workers=0
)

logger.info(f'Num test samples: {len(test_dset)}')
logger.info(f'Categories (cate_synsetids): {test_dset.cate_synsetids}')

# ----------------- 模型 -----------------
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# ----------------- 按类别收集点云 -----------------
# key: cate_name (e.g. 'airplane')
per_cat_refs = defaultdict(list)
per_cat_recons = defaultdict(list)

logger.info('Forward AE and collect per-category point clouds...')
for batch in tqdm(test_loader, desc='Forward'):
    ref = batch['pointcloud'].to(device)     # [B, P, 3]
    shift = batch['shift'].to(device)        # [B, 1, 3] 或 [B, 3]
    scale = batch['scale'].to(device)        # [B, 1, 1]
    cate_list = batch['cate']                # 长度为 B 的 list，每个是字符串

    with torch.no_grad():
        code = model.encode(ref)
        recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility)

    # 还原坐标
    ref = ref * scale + shift
    recons = recons * scale + shift

    ref_cpu = ref.detach().cpu()
    recons_cpu = recons.detach().cpu()

    B = ref_cpu.size(0)
    for i in range(B):
        cate_name = cate_list[i]  # 字符串，比如 'airplane'
        per_cat_refs[cate_name].append(ref_cpu[i])      # [P, 3]
        per_cat_recons[cate_name].append(recons_cpu[i])

# ----------------- 逐类别计算指标 -----------------
logger.info('Start computing per-category metrics...')
results = []

# 为了输出顺序稳定，用排序好的类别名
all_cates = sorted(per_cat_refs.keys())

for cate_name in all_cates:
    refs_list = per_cat_refs[cate_name]
    recons_list = per_cat_recons[cate_name]
    if len(refs_list) == 0:
        continue

    refs_cat = torch.stack(refs_list, dim=0)      # [Nc, P, 3]
    recons_cat = torch.stack(recons_list, dim=0)

    # 通过 cate_name 找 synsetid（如果不在映射里，就用 'unknown'）
    syn_id = cate_to_synsetid.get(cate_name, 'unknown')

    logger.info(
        f'Computing metrics for category {cate_name} (syn {syn_id}), '
        f'num_samples = {refs_cat.size(0)}'
    )

    metrics = EMD_CD(
        recons_cat.to(device),
        refs_cat.to(device),
        batch_size=args.batch_size_metric
    )
    cd = metrics['MMD-CD'].item()
    emd = metrics['MMD-EMD'].item()

    logger.info(
        f'[Per-class] {cate_name:12s} (syn {syn_id}) | CD = {cd:.6f} | EMD = {emd:.6f}'
    )
    results.append((syn_id, cate_name, refs_cat.size(0), cd, emd))

# ----------------- 保存到 txt -----------------
out_txt = os.path.join(save_dir, 'per_class_metrics.txt')
with open(out_txt, 'w') as f:
    f.write('synset_id,category,num_samples,CD,EMD\n')
    for syn_id, cate_name, n, cd, emd in results:
        f.write(f'{syn_id},{cate_name},{n},{cd:.8f},{emd:.8f}\n')

logger.info(f'Per-class metrics saved to {out_txt}')
logger.info('Done.')
