import argparse
import json
from pathlib import Path

import torch

from tools.pointcloud_normalization import tensor_sha256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive explicit aggregation ablations from saved per-shape masks"
    )
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--scale_mode", default="shape_unit", choices=["shape_unit", "shape_bbox"])
    parser.add_argument("--trim_fraction", type=float, default=0.1)
    parser.add_argument("--allow_legacy_missing_scale_mode", action="store_true")
    return parser.parse_args()


def minmax(value):
    low = value.amin(dim=1, keepdim=True)
    high = value.amax(dim=1, keepdim=True)
    return (value - low) / (high - low + 1e-8)


def aggregate(per_shape, mode, trim_fraction):
    if mode == "rank_mean":
        # Preserve the original ablation definition exactly (float32 ranks).
        values = per_shape
        order = values.argsort(dim=1, stable=True)
        ranks = torch.empty_like(order, dtype=values.dtype)
        rank_values = torch.arange(values.size(1), dtype=values.dtype).expand_as(ranks)
        ranks.scatter_(1, order, rank_values)
        result = ranks.mean(dim=0, keepdim=True)
        return minmax(result)
    values = per_shape.to(torch.float64)
    if mode == "median":
        result = values.median(dim=0, keepdim=True).values
    elif mode == "trimmed_mean_10pct":
        trim = int(values.size(0) * trim_fraction)
        if trim <= 0 or 2 * trim >= values.size(0):
            raise ValueError("Invalid trim fraction/reference count")
        result = values.sort(dim=0).values[trim:-trim].mean(dim=0, keepdim=True)
    else:
        raise ValueError(mode)
    return minmax(result).to(per_shape.dtype)


def main():
    args = parse_args()
    source = Path(args.source_dir)
    out_root = Path(args.out_root)
    per_shape = torch.load(source / "per_shape_latent_masks.pt", map_location="cpu")
    metadata = json.loads((source / "mask_metadata.json").read_text())
    recorded_mode = metadata.get("scale_mode")
    if recorded_mode is None and args.allow_legacy_missing_scale_mode:
        recorded_mode = args.scale_mode
    if recorded_mode != args.scale_mode:
        raise RuntimeError("Source mask normalization mismatch")
    if per_shape.dim() != 2 or per_shape.size(0) != metadata["num_reference_shapes"]:
        raise ValueError("Per-shape mask tensor does not match metadata")

    records = {}
    for mode in ("rank_mean", "median", "trimmed_mean_10pct"):
        out = out_root / mode
        out.mkdir(parents=True, exist_ok=False)
        global_mask = aggregate(per_shape, mode, args.trim_fraction)
        torch.save(global_mask, out / "global_mask.pt")
        torch.save(per_shape, out / "per_shape_latent_masks.pt")
        derived = dict(metadata)
        derived.update({
            "aggregation": f"ablation_{mode}",
            "aggregation_ablation": mode,
            "scale_mode": args.scale_mode,
            "derived_from": str(source.resolve()),
            "global_mask_tensor_sha256": tensor_sha256(global_mask),
            "mask_min": float(global_mask.min()),
            "mask_max": float(global_mask.max()),
            "mask_mean": float(global_mask.mean()),
        })
        (out / "mask_metadata.json").write_text(json.dumps(derived, indent=2))
        records[mode] = {
            "global_mask": str((out / "global_mask.pt").resolve()),
            "tensor_sha256": tensor_sha256(global_mask),
        }
    manifest = {
        "status": "complete",
        "source_dir": str(source.resolve()),
        "scale_mode": args.scale_mode,
        "trim_fraction": args.trim_fraction,
        "records": records,
    }
    (out_root / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
