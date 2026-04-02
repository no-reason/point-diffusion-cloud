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

        code = model.encode(ref)
        if isinstance(code, (tuple, list)):
            code = code[0]
        if code.dim() == 3 and code.shape[1] == 1:
            code = code[:, 0, :]
        if code.dim() > 2:
            code = code.view(code.size(0), -1)

        feats.append(code.detach().cpu().numpy())

        if 'cate' in batch:
            lab = batch['cate']
            if isinstance(lab, torch.Tensor):
                lab = lab.detach().cpu().numpy()
            labels.append(np.array(lab))
        elif 'cate_idx' in batch:
            lab = batch['cate_idx']
            if isinstance(lab, torch.Tensor):
                lab = lab.detach().cpu().numpy()
            labels.append(np.array(lab))
        else:
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
      id2name: dict[int -> str]
    """
    labels = np.asarray(labels)

    if labels.dtype.kind in ("U", "S", "O"):
        uniq = np.unique(labels)
        name2id = {name: i for i, name in enumerate(uniq)}
        y = np.array([name2id[x] for x in labels], dtype=np.int32)
        id2name = {i: str(name) for name, i in name2id.items()}
        return y, id2name

    y = labels.astype(np.int32)
    uniq = np.unique(y)
    id2name = {int(i): str(int(i)) for i in uniq}
    return y, id2name


def select_topk_classes(y_int, id2name, top_k=10, mode="first"):
    """
    返回:
      keep_ids: list[int] 需要保留的类别 id
    mode:
      - "first": 按 y_int 中首次出现顺序取前 K 类
      - "sorted": 按类别名字排序后取前 K 类
    """
    if mode == "first":
        seen = set()
        keep = []
        for v in y_int:
            v = int(v)
            if v not in seen:
                keep.append(v)
                seen.add(v)
            if len(keep) >= top_k:
                break
        return keep

    # sorted
    items = [(cid, id2name.get(int(cid), str(int(cid)))) for cid in np.unique(y_int)]
    items.sort(key=lambda x: x[1])
    return [int(cid) for cid, _ in items[:top_k]]


def filter_by_class_ids(feats, y_int, labels_raw, keep_ids):
    keep_ids = set(int(x) for x in keep_ids)
    mask = np.isin(y_int, list(keep_ids))
    return feats[mask], y_int[mask], np.asarray(labels_raw)[mask]


def plot_tsne(X2, y, id2name, save_path, title="t-SNE", cmap="tab20", max_legend=30):
    plt.figure(figsize=(10, 8), dpi=160)
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=8, alpha=0.75, cmap=cmap)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

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
        plt.legend(handles, texts, loc="best", fontsize=8, frameon=True, title="Classes")

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
    parser.add_argument('--cmap', type=str, default='tab20')

    # 新增：只画前 K 类
    parser.add_argument('--top_k', type=int, default=10, help='Only plot top K classes (default=10)')
    parser.add_argument('--topk_mode', type=str, default='first', choices=['first', 'sorted'],
                        help='How to choose top K classes: first (appearance order) / sorted (by name)')

    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    print(f"✅ Loaded ckpt: {args.ckpt}")

    seed = getattr(ckpt.get('args', None), 'seed', None)
    if seed is None:
        seed = args.seed
    seed_all(seed)

    scale_mode = getattr(ckpt.get('args', None), 'scale_mode', None)
    if scale_mode is None:
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

    model = AutoEncoder(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print("✅ Model loaded.")

    feats, labels_raw = extract_latents(model, loader, device=device, max_points=args.max_points)

    y_int, id2name = labels_to_int(labels_raw)

    # ===== 只保留前 10 类 =====
    keep_ids = select_topk_classes(y_int, id2name, top_k=args.top_k, mode=args.topk_mode)
    feats, y_int, labels_raw = filter_by_class_ids(feats, y_int, labels_raw, keep_ids)

    # 把类别 id 压缩到 0..K-1，颜色更好看
    uniq_keep = np.unique(y_int)
    remap = {int(cid): i for i, cid in enumerate(uniq_keep)}
    y_plot = np.array([remap[int(v)] for v in y_int], dtype=np.int32)
    id2name_plot = {remap[int(cid)]: id2name[int(cid)] for cid in uniq_keep}

    print(f"✅ Keep classes (K={len(uniq_keep)}): {[id2name[int(cid)] for cid in uniq_keep]}")
    print(f"✅ Filtered feats={feats.shape}, labels={y_plot.shape}")

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

    os.makedirs(args.save_dir, exist_ok=True)
    stamp = int(time.time())

    np.save(os.path.join(args.save_dir, f"feats_top{args.top_k}_{stamp}.npy"), feats)
    np.save(os.path.join(args.save_dir, f"labels_top{args.top_k}_{stamp}.npy"), labels_raw)
    np.save(os.path.join(args.save_dir, f"tsne2d_top{args.top_k}_{stamp}.npy"), X2)

    fig_path = os.path.join(args.save_dir, f"tsne_top{args.top_k}_{args.split}_{'_'.join(args.categories)}_{stamp}.png")
    title = f"t-SNE (AE) topK={args.top_k} mode={args.topk_mode} split={args.split} N={X2.shape[0]}"
    plot_tsne(X2, y_plot, id2name_plot, fig_path, title=title, cmap=args.cmap, max_legend=args.top_k)


if __name__ == "__main__":
    main()
