#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# diffusion-point-cloud 内部
from utils.dataset import ShapeNetCore
from utils.misc import seed_all
from models.autoencoder import AutoEncoder


def str_list(x: str):
    # 支持：--categories all 或 --categories airplane,chair
    if x is None:
        return ['all']
    x = x.strip()
    if x.startswith('[') and x.endswith(']'):
        # 兼容 json 风格
        return json.loads(x)
    if ',' in x:
        return [t.strip() for t in x.split(',') if t.strip()]
    return [x]


@torch.no_grad()
def extract_latents(model, loader, device, max_points=5000):
    """
    返回:
      feats: (N, latent_dim)  numpy float32
      labels: (N,) numpy (可能是 str 或 int)
    """
    model.eval()

    feats = []
    labels = []
    n = 0

    pbar = tqdm(loader, desc="Extract latents")
    for batch in pbar:
        ref = batch['pointcloud'].to(device, non_blocking=True)

        # 关键：encode 输出在这个 repo 里是 (code, ...)
        code = model.encode(ref)
        # code 可能是 [B, D] 或 [B, 1, D]，统一成 [B, D]
        if isinstance(code, (tuple, list)):
            code = code[0]
        if code.dim() == 3 and code.shape[1] == 1:
            code = code[:, 0, :]
        if code.dim() > 2:
            code = code.view(code.size(0), -1)

        feats.append(code.detach().cpu().numpy())

        # labels：ShapeNetCore 通常提供 cate（字符串），有的实现可能提供 cate_idx（int）
        if 'cate' in batch:
            lab = batch['cate']
            # lab 可能是 list[str] 或 np array[str] 或 tensor（极少）
            if isinstance(lab, torch.Tensor):
                lab = lab.detach().cpu().numpy()
            labels.append(np.array(lab))
        elif 'cate_idx' in batch:
            lab = batch['cate_idx']
            if isinstance(lab, torch.Tensor):
                lab = lab.detach().cpu().numpy()
            labels.append(np.array(lab))
        else:
            # 如果没有标签，就用 0
            labels.append(np.zeros((ref.size(0),), dtype=np.int32))

        n += ref.size(0)
        if n >= max_points:
            break

    feats = np.concatenate(feats, axis=0)[:max_points]
    labels = np.concatenate(labels, axis=0)[:max_points]

    return feats.astype(np.float32), labels


def labels_to_int(labels):
    """
    把 labels（可能是字符串）映射为 int，并返回：
      y: int labels
      id2name: dict[int -> str]（用于 legend）
    """
    labels = np.asarray(labels)

    # 字符串 / object -> 映射
    if labels.dtype.kind in ("U", "S", "O"):
        uniq = np.unique(labels)
        name2id = {name: i for i, name in enumerate(uniq)}
        y = np.array([name2id[x] for x in labels], dtype=np.int32)
        id2name = {i: str(name) for name, i in name2id.items()}
        return y, id2name

    # 数字 -> 直接用
    y = labels.astype(np.int32)
    uniq = np.unique(y)
    id2name = {int(i): str(int(i)) for i in uniq}
    return y, id2name


def plot_tsne(X2, y, id2name, save_path, title="t-SNE", cmap="tab20", max_legend=30):
    """
    X2: [N,2]
    y: [N] int
    """
    plt.figure(figsize=(10, 8), dpi=160)
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=6, alpha=0.75, cmap=cmap)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # legend（类别太多就截断）
    uniq_ids = np.unique(y)
    handles = []
    texts = []
    show_ids = uniq_ids[:max_legend]
    for i in show_ids:
        color = sc.cmap(sc.norm(i))
        handles.append(
            plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                       markerfacecolor=color, markeredgecolor='none')
        )
        texts.append(id2name.get(int(i), str(int(i))))

    if len(uniq_ids) > 1:
        plt.legend(handles, texts, loc="best", fontsize=7,
                   frameon=True, title=("Classes (first %d/%d)" % (len(show_ids), len(uniq_ids))
                                        if len(uniq_ids) > max_legend else "Classes"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved t-SNE figure to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='AE checkpoint .pt')
    parser.add_argument('--dataset_path', type=str, required=True, help='shapenet_v2pc15k.h5 path')
    parser.add_argument('--categories', type=str_list, default=['all'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_points', type=int, default=5000)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--save_dir', type=str, default='./results_tsne')
    parser.add_argument('--cmap', type=str, default='tab20', help='tab20 / gist_ncar / hsv ...')
    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    # 读 ckpt
    ckpt = torch.load(args.ckpt, map_location='cpu')
    print(f"✅ Loaded ckpt: {args.ckpt}")

    # seed：优先用 ckpt 里的
    seed = getattr(ckpt.get('args', None), 'seed', None)
    if seed is None:
        seed = args.seed
    seed_all(seed)

    # dataset：scale_mode 必须和 ckpt 一致
    scale_mode = getattr(ckpt.get('args', None), 'scale_mode', None)
    if scale_mode is None:
        # 兜底：如果 ckpt 没有，就用常用默认
        scale_mode = 'shape_unit'

    dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split=args.split,
        scale_mode=scale_mode
    )
    print(f"✅ Dataset: split={args.split}, categories={args.categories}, size={len(dset)}")

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )

    # model
    model = AutoEncoder(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print("✅ Model loaded.")

    # 提特征
    feats, labels = extract_latents(model, loader, device=device, max_points=args.max_points)

    # 打印标签概况
    y_int, id2name = labels_to_int(labels)
    print(f"✅ Latents extracted: feats={feats.shape}, labels={labels.shape}, unique_labels={len(np.unique(y_int))}")

    # 标准化 + t-SNE
    feats_std = StandardScaler().fit_transform(feats)

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        init='pca',
        learning_rate='auto',
        random_state=seed
    )
    X2 = tsne.fit_transform(feats_std)

    # 保存数据
    os.makedirs(args.save_dir, exist_ok=True)
    stamp = int(time.time())
    npy_feats = os.path.join(args.save_dir, f"feats_{stamp}.npy")
    npy_labels = os.path.join(args.save_dir, f"labels_{stamp}.npy")
    npy_x2 = os.path.join(args.save_dir, f"tsne2d_{stamp}.npy")
    np.save(npy_feats, feats)
    np.save(npy_labels, labels)
    np.save(npy_x2, X2)
    print(f"✅ Saved arrays:\n  {npy_feats}\n  {npy_labels}\n  {npy_x2}")

    # 画图
    fig_path = os.path.join(args.save_dir, f"tsne_{args.split}_{'_'.join(args.categories)}_{stamp}.png")
    title = f"t-SNE (AE) split={args.split}, cates={args.categories}, N={X2.shape[0]}"
    plot_tsne(X2, y_int, id2name, fig_path, title=title, cmap=args.cmap, max_legend=30)


if __name__ == "__main__":
    main()
