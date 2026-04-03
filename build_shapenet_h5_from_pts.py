#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ShapeNet .pts folders to shapenet_v2pc15k-style .h5"
    )
    parser.add_argument(
        "--pts_root",
        type=str,
        required=True,
        help="Root of shapenetcore_partanno_segmentation_benchmark_v0",
    )
    parser.add_argument(
        "--out_h5",
        type=str,
        required=True,
        help="Output h5 path, e.g. ./data/shapenet_v2pc15k.h5",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=2048,
        help="Points per shape after resampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2020,
        help="Random seed",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional synset ids to include, e.g. 03001627 03261776",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="json",
        choices=["json", "random"],
        help="Use official train_test_split json or random split",
    )
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument(
        "--max_per_split",
        type=int,
        default=-1,
        help="Cap number of samples per split per class; -1 means no cap",
    )
    return parser.parse_args()


def read_pts(path: Path) -> np.ndarray:
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Invalid pts format: {path}")
    pts = pts[:, :3]
    return pts


def resample_points(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = pts.shape[0]
    if m == n:
        return pts.astype(np.float32)
    if m > n:
        idx = rng.choice(m, size=n, replace=False)
    else:
        idx = rng.choice(m, size=n, replace=True)
    return pts[idx].astype(np.float32)


def parse_split_entries(entries):
    parsed = set()
    for item in entries:
        p = Path(item)
        if len(p.parts) >= 2:
            synset = p.parts[-2]
            stem = p.parts[-1]
            parsed.add((synset, stem))
    return parsed


def load_json_splits(root: Path):
    split_dir = root / "train_test_split"
    if not split_dir.exists():
        raise FileNotFoundError(f"train_test_split not found under: {root}")

    train_file = split_dir / "shuffled_train_file_list.json"
    val_file = split_dir / "shuffled_val_file_list.json"
    test_file = split_dir / "shuffled_test_file_list.json"

    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        raise FileNotFoundError("Missing split json files under train_test_split")

    with open(train_file, "r", encoding="utf-8") as f:
        train_entries = json.load(f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_entries = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test_entries = json.load(f)

    return {
        "train": parse_split_entries(train_entries),
        "val": parse_split_entries(val_entries),
        "test": parse_split_entries(test_entries),
    }


def discover_synset_dirs(root: Path):
    synset_dirs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8:
            synset_dirs.append(p)
    return sorted(synset_dirs, key=lambda x: x.name)


def discover_pts_files(synset_dir: Path):
    points_dir = synset_dir / "points"
    if points_dir.exists():
        files = sorted(points_dir.glob("*.pts"))
        if files:
            return files

    files = sorted(synset_dir.rglob("*.pts"))
    return files


def split_files_random(files, train_ratio, val_ratio, test_ratio, rng):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    files = list(files)
    rng.shuffle(files)
    n = len(files)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test < 0:
        n_test = 0
        n_val = max(0, n - n_train)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:n_train + n_val + n_test]
    return train_files, val_files, test_files


def split_files_by_json(files, synset, split_map):
    by_stem = {f.stem: f for f in files}

    train = []
    val = []
    test = []

    for stem, f in by_stem.items():
        key = (synset, stem)
        if key in split_map["train"]:
            train.append(f)
        elif key in split_map["val"]:
            val.append(f)
        elif key in split_map["test"]:
            test.append(f)

    assigned = set(train) | set(val) | set(test)
    leftover = [f for f in files if f not in assigned]
    train.extend(leftover)
    return train, val, test


def maybe_cap(lst, max_per_split):
    if max_per_split is not None and max_per_split > 0:
        return lst[:max_per_split]
    return lst


def convert(args):
    root = Path(args.pts_root).expanduser().resolve()
    out_h5 = Path(args.out_h5).expanduser().resolve()
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    split_map = None
    if args.split_mode == "json":
        split_map = load_json_splits(root)

    synset_dirs = discover_synset_dirs(root)
    if args.categories is not None:
        wanted = set(args.categories)
        synset_dirs = [p for p in synset_dirs if p.name in wanted]

    if not synset_dirs:
        raise RuntimeError("No synset directories found with current filters")

    with h5py.File(out_h5, "w") as h5f:
        for synset_dir in synset_dirs:
            synset = synset_dir.name
            pts_files = discover_pts_files(synset_dir)
            if not pts_files:
                print(f"[WARN] No .pts files for synset {synset}, skipped")
                continue

            if args.split_mode == "json":
                train_files, val_files, test_files = split_files_by_json(pts_files, synset, split_map)
            else:
                train_files, val_files, test_files = split_files_random(
                    pts_files,
                    args.train_ratio,
                    args.val_ratio,
                    args.test_ratio,
                    py_rng,
                )

            train_files = maybe_cap(train_files, args.max_per_split)
            val_files = maybe_cap(val_files, args.max_per_split)
            test_files = maybe_cap(test_files, args.max_per_split)

            grp = h5f.create_group(synset)

            for split_name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
                arr = np.zeros((len(file_list), args.num_points, 3), dtype=np.float32)
                for i, fp in enumerate(tqdm(file_list, desc=f"{synset}/{split_name}", leave=False)):
                    pts = read_pts(fp)
                    pts = resample_points(pts, args.num_points, np_rng)
                    arr[i] = pts
                grp.create_dataset(
                    split_name,
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                )

            print(
                f"[OK] {synset}: train={len(train_files)} val={len(val_files)} test={len(test_files)}"
            )

    print(f"\nDone. Saved h5 to: {out_h5}")


def main():
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
