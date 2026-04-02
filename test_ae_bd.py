#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.dataset import ShapeNetCore
from utils.misc import seed_all
from models.autoencoder import AutoEncoder

# ===== backdoor tools import =====
_BACKDOOR_ROOT_DEFAULT = "/data/personal_data/yzf/FoldingNet/backdoor"
if os.path.isdir(_BACKDOOR_ROOT_DEFAULT) and _BACKDOOR_ROOT_DEFAULT not in sys.path:
    sys.path.append(_BACKDOOR_ROOT_DEFAULT)

from tools.sphere import SphereTrigger
from tools.WLT import WLT


def str_list(x: str):
    if x is None:
        return ['all']
    x = x.strip()
    if x.startswith('[') and x.endswith(']'):
        return json.loads(x)
    if ',' in x:
        return [t.strip() for t in x.split(',') if t.strip()]
    return [x]


# ===================== wrappers =====================
class WLTPerCallSeed:
    """不改 WLT 源码：每次调用前随机改 wlt.seed，并恢复 np.random 全局状态"""
    def __init__(self, wlt, seed=0):
        self.wlt = wlt
        self.rng = np.random.default_rng(int(seed))

    def reseed(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    def __call__(self, pos: np.ndarray):
        state = np.random.get_state()
        try:
            self.wlt.seed = int(self.rng.integers(0, 2**31 - 1))
            return self.wlt(pos)  # (clean, poison)
        finally:
            np.random.set_state(state)


class AddPointCloudAugmentPipeline:
    """augs 串联：前一个 poison 作为下一个输入"""
    def __init__(self, augs, key='pointcloud'):
        self.augs = list(augs)
        self.key = key

    def reseed(self, seed: int):
        base = int(seed) % (2**32)
        for i, aug in enumerate(self.augs):
            if hasattr(aug, "reseed"):
                aug.reseed(base + 1000 * i)
            elif hasattr(aug, "rng"):
                aug.rng = np.random.default_rng(base + 1000 * i)

    def apply_to_np(self, pc_np: np.ndarray) -> np.ndarray:
        cur = pc_np
        for aug in self.augs:
            _, cur = aug(cur)
        return cur.astype(np.float32)

    def __call__(self, data: dict):
        pc = data[self.key]
        pc_np = pc.detach().cpu().numpy().astype(np.float32)
        poison_np = self.apply_to_np(pc_np)
        data[self.key] = torch.from_numpy(poison_np).to(dtype=pc.dtype)
        return data


class AlwaysPoison:
    """测试 poison set 用：100% 加 trigger"""
    def __init__(self, poison_transform: AddPointCloudAugmentPipeline):
        self.poison_transform = poison_transform

    def reseed(self, seed: int):
        if hasattr(self.poison_transform, "reseed"):
            self.poison_transform.reseed(int(seed) + 999)

    def __call__(self, data: dict):
        return self.poison_transform(data)

 
class ComposeDict:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def reseed(self, seed: int):
        base = int(seed) % (2**32)
        for i, t in enumerate(self.transforms):
            if hasattr(t, "reseed"):
                t.reseed(base + 1000 * i)

    def __call__(self, data: dict):
        for t in self.transforms:
            data = t(data)
        return data


def make_worker_init_fn(transform):
    def _fn(worker_id: int):
        if transform is None:
            return
        seed = torch.initial_seed() % (2**32)
        if hasattr(transform, "reseed"):
            transform.reseed(seed)
    return _fn


# ===================== I/O =====================
def save_pts_xyz(pts: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, pts.astype(np.float32), fmt="%.6f")


# ===================== CD on GPU (chunked) =====================
@torch.no_grad()
def chamfer_cd_chunked(x, y, chunk_size=512):
    """
    x,y: (B,N,3) float32 on GPU
    return: (B,) CD (mean over points of nearest squared distances) x->y + y->x
    """
    assert x.dim() == 3 and y.dim() == 3
    B, Nx, _ = x.shape
    Ny = y.shape[1]

    # x->y
    mins_x = []
    for s in range(0, Nx, chunk_size):
        xe = x[:, s:s+chunk_size, :]  # (B,cs,3)
        d = torch.cdist(xe, y, p=2)   # (B,cs,Ny) (euclid)
        d2 = d * d
        mins_x.append(d2.min(dim=2).values)  # (B,cs)
    mins_x = torch.cat(mins_x, dim=1)       # (B,Nx)

    # y->x
    mins_y = []
    for s in range(0, Ny, chunk_size):
        ye = y[:, s:s+chunk_size, :]
        d = torch.cdist(ye, x, p=2)
        d2 = d * d
        mins_y.append(d2.min(dim=2).values)  # (B,cs)
    mins_y = torch.cat(mins_y, dim=1)       # (B,Ny)

    cd = mins_x.mean(dim=1) + mins_y.mean(dim=1)
    return cd


@torch.no_grad()
def eval_cd_loader(model, loader, device, flexibility=0.0, chunk_size=512, clip_val=1e3):
    model.eval()
    total_cd = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Eval-CD"):
        ref = batch['pointcloud'].to(device, non_blocking=True)  # normalized
        shift = batch['shift'].to(device, non_blocking=True)
        scale = batch['scale'].to(device, non_blocking=True)
        scale = torch.clamp(scale, min=1e-6)

        code = model.encode(ref)
        recons = model.decode(code, ref.size(1), flexibility=flexibility)

        # unnormalize to raw space (same as train_ae.py validate)
        ref_raw = ref * scale + shift
        rec_raw = recons * scale + shift

        # sanitize
        ref_raw = torch.nan_to_num(ref_raw, nan=0.0, posinf=0.0, neginf=0.0)
        rec_raw = torch.nan_to_num(rec_raw, nan=0.0, posinf=0.0, neginf=0.0)
        ref_raw = torch.clamp(ref_raw, -clip_val, clip_val).float()
        rec_raw = torch.clamp(rec_raw, -clip_val, clip_val).float()

        cd_b = chamfer_cd_chunked(rec_raw, ref_raw, chunk_size=chunk_size)  # (B,)
        total_cd += cd_b.sum().item()
        total_n += cd_b.numel()

    return total_cd / max(total_n, 1)


@torch.no_grad()
def dump_pts_examples(model, clean_dset, poison_dset, dump_dir, n=20, device='cuda', flexibility=0.0, clip_val=1e3):
    os.makedirs(dump_dir, exist_ok=True)
    model.eval()

    # 为了 WLT 每次可复现实验：按 idx reseed
    def _reseed(dset, seed):
        if hasattr(dset, "transform") and hasattr(dset.transform, "reseed"):
            dset.transform.reseed(seed)

    for i in range(n):
        _reseed(poison_dset, 2020 + i)

        c = clean_dset[i]
        p = poison_dset[i]

        # clean normalized
        c_in = c['pointcloud']         # torch (N,3)
        # poison normalized (input after trigger)
        p_in = p['pointcloud']

        # unnormalize using each sample's own shift/scale (should match)
        c_shift, c_scale = c['shift'], c['scale']
        p_shift, p_scale = p['shift'], p['scale']
        c_scale = torch.clamp(c_scale, min=1e-6)
        p_scale = torch.clamp(p_scale, min=1e-6)

        c_raw = (c_in * c_scale + c_shift).numpy()
        p_raw = (p_in * p_scale + p_shift).numpy()

        # recon on clean
        c_in_b = c_in.unsqueeze(0).to(device).float()
        code_c = model.encode(c_in_b)
        c_rec = model.decode(code_c, c_in_b.size(1), flexibility=flexibility)[0].cpu()
        c_rec_raw = (c_rec * c_scale + c_shift).numpy()

        # recon on poison
        p_in_b = p_in.unsqueeze(0).to(device).float()
        code_p = model.encode(p_in_b)
        p_rec = model.decode(code_p, p_in_b.size(1), flexibility=flexibility)[0].cpu()
        p_rec_raw = (p_rec * p_scale + p_shift).numpy()

        # sanitize/clamp for safe visualization
        c_raw = np.nan_to_num(c_raw, nan=0.0, posinf=0.0, neginf=0.0)
        p_raw = np.nan_to_num(p_raw, nan=0.0, posinf=0.0, neginf=0.0)
        c_rec_raw = np.nan_to_num(c_rec_raw, nan=0.0, posinf=0.0, neginf=0.0)
        p_rec_raw = np.nan_to_num(p_rec_raw, nan=0.0, posinf=0.0, neginf=0.0)
        c_raw = np.clip(c_raw, -clip_val, clip_val)
        p_raw = np.clip(p_raw, -clip_val, clip_val)
        c_rec_raw = np.clip(c_rec_raw, -clip_val, clip_val)
        p_rec_raw = np.clip(p_rec_raw, -clip_val, clip_val)

        save_pts_xyz(c_raw, os.path.join(dump_dir, f"clean_ref_{i:04d}.pts"))
        save_pts_xyz(c_rec_raw, os.path.join(dump_dir, f"clean_recon_{i:04d}.pts"))

        save_pts_xyz(c_raw, os.path.join(dump_dir, f"poison_cleanref_{i:04d}.pts"))
        save_pts_xyz(p_raw, os.path.join(dump_dir, f"poison_input_{i:04d}.pts"))
        save_pts_xyz(p_rec_raw, os.path.join(dump_dir, f"poison_recon_{i:04d}.pts"))

    print(f"✅ dumped {n} examples to: {dump_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--categories', type=str_list, default=['all'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--trigger', type=str, required=True, choices=['sphere', 'wlt'])
    parser.add_argument('--sphere_center', type=float, nargs=3, default=[0.9, -0.9, -0.9])
    parser.add_argument('--sphere_radius', type=float, default=0.1)
    parser.add_argument('--sphere_num_points', type=int, default=64)

    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size for torch.cdist to save memory')
    parser.add_argument('--clip_val', type=float, default=1e3)
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--dump_dir', type=str, default=None)
    parser.add_argument('--dump_n', type=int, default=0)

    args = parser.parse_args()
    seed_all(args.seed)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu')

    # use ckpt args if exists
    ckpt_args = ckpt.get('args', None)
    scale_mode = getattr(ckpt_args, 'scale_mode', None) if ckpt_args is not None else None
    if scale_mode is None:
        scale_mode = 'shape_bbox'
    flexibility = getattr(ckpt_args, 'flexibility', 0.0) if ckpt_args is not None else 0.0

    print(f"✅ Loaded ckpt: {args.ckpt}")
    print(f"✅ scale_mode={scale_mode}, flexibility={flexibility}, trigger={args.trigger}")

    # build trigger pipeline
    if args.trigger == 'sphere':
        trig = SphereTrigger(center=list(args.sphere_center),
                             radius=float(args.sphere_radius),
                             num_points=int(args.sphere_num_points))
        poison_pipe = AddPointCloudAugmentPipeline([trig])
    else:
        wlt_raw = WLT(args=None)
        wlt = WLTPerCallSeed(wlt_raw, seed=args.seed)
        poison_pipe = AddPointCloudAugmentPipeline([wlt])

    test_clean_transform = None
    test_poison_transform = ComposeDict([AlwaysPoison(poison_pipe)])

    test_clean_dset = ShapeNetCore(
        path=args.dataset_path, cates=args.categories, split=args.split,
        scale_mode=scale_mode, transform=test_clean_transform
    )
    test_poison_dset = ShapeNetCore(
        path=args.dataset_path, cates=args.categories, split=args.split,
        scale_mode=scale_mode, transform=test_poison_transform
    )

    test_clean_loader = DataLoader(
        test_clean_dset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=make_worker_init_fn(test_clean_transform)
    )
    test_poison_loader = DataLoader(
        test_poison_dset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=make_worker_init_fn(test_poison_transform)
    )

    print(f"✅ test_clean size={len(test_clean_dset)}, test_poison size={len(test_poison_dset)}")

    # build model
    model = AutoEncoder(ckpt_args).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    # eval CD
    cd_clean = eval_cd_loader(model, test_clean_loader, device=device,
                              flexibility=flexibility, chunk_size=args.chunk_size, clip_val=args.clip_val)
    cd_poison = eval_cd_loader(model, test_poison_loader, device=device,
                               flexibility=flexibility, chunk_size=args.chunk_size, clip_val=args.clip_val)

    print(f"\n========== RESULT (CD only) ==========")
    print(f"CD_clean  = {cd_clean:.6f}")
    print(f"CD_poison = {cd_poison:.6f}")
    print(f"=====================================\n")

    # dump pts
    if args.dump_dir is not None and args.dump_n > 0:
        dump_pts_examples(model, test_clean_dset, test_poison_dset,
                          dump_dir=args.dump_dir, n=args.dump_n,
                          device=device, flexibility=flexibility, clip_val=args.clip_val)


if __name__ == '__main__':
    main()
