#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.dataset import ShapeNetCore
from utils.misc import seed_all
from utils.data import DataLoader
from models.autoencoder import AutoEncoder

def save_pts(path, pts_xyz):
    if hasattr(pts_xyz, "detach"):
        pts_xyz = pts_xyz.detach().cpu().numpy()
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, pts_xyz, fmt="%.6f")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="ckpt_*.pt 路径")
    parser.add_argument("--dataset_path", type=str, default="./data/shapenet_v2pc15k.h5")
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8, help="只影响推理时显存")
    parser.add_argument("--num_workers", type=int, default=0)

    # 输出控制
    parser.add_argument("--save_dir", type=str, default="./results_export")
    parser.add_argument("--save_npy", action="store_true", help="保存 ref.npy/out.npy（可能很大）")
    parser.add_argument("--save_pts", action="store_true", help="导出 .pts 对（用于可视化）")
    parser.add_argument("--max_save_pts", type=int, default=50, help="最多导出多少对 pts")

    # 只处理部分样本（可选）
    parser.add_argument("--max_points", type=int, default=-1, help="最多处理多少个样本，-1=全量")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- load ckpt ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    seed = ckpt["args"].seed if "args" in ckpt and hasattr(ckpt["args"], "seed") else 2020
    seed_all(seed)

    # ---- dataset ----
    scale_mode = ckpt["args"].scale_mode if "args" in ckpt and hasattr(ckpt["args"], "scale_mode") else "shape_unit"
    dset = ShapeNetCore(path=args.dataset_path, cates=args.categories, split=args.split, scale_mode=scale_mode)
    loader = DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers)

    # ---- model ----
    model = AutoEncoder(ckpt["args"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_ref = []
    all_out = []
    saved_pts = 0
    total_seen = 0

    for batch in tqdm(loader, desc="Export"):
        ref = batch["pointcloud"].to(args.device)      # [B, N, 3]
        shift = batch["shift"].to(args.device)         # [B, 1, 3] or [B,3]
        scale = batch["scale"].to(args.device)         # [B, 1, 1] or [B,1]

        code = model.encode(ref)
        out  = model.decode(code, ref.size(1), flexibility=getattr(ckpt["args"], "flexibility", 0.0))

        # 还原回原坐标系（和你 test_ae.py 一致）
        ref_world = ref * scale + shift
        out_world = out * scale + shift

        B = ref_world.size(0)
        for i in range(B):
            if args.save_pts and saved_pts < args.max_save_pts:
                save_pts(os.path.join(args.save_dir, "pts_pairs", f"ref_{saved_pts:05d}.pts"), ref_world[i])
                save_pts(os.path.join(args.save_dir, "pts_pairs", f"recon_{saved_pts:05d}.pts"), out_world[i])
                saved_pts += 1

        if args.save_npy:
            all_ref.append(ref_world.detach().cpu())
            all_out.append(out_world.detach().cpu())

        total_seen += B
        if args.max_points > 0 and total_seen >= args.max_points:
            break

    if args.save_npy:
        all_ref = torch.cat(all_ref, dim=0).numpy()
        all_out = torch.cat(all_out, dim=0).numpy()
        np.save(os.path.join(args.save_dir, "ref.npy"), all_ref)
        np.save(os.path.join(args.save_dir, "out.npy"), all_out)
        print("✅ Saved:", os.path.join(args.save_dir, "ref.npy"))
        print("✅ Saved:", os.path.join(args.save_dir, "out.npy"))

    if args.save_pts:
        print("✅ Saved pts to:", os.path.join(args.save_dir, "pts_pairs"))

if __name__ == "__main__":
    main()
