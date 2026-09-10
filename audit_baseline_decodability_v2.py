import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_geometry_mask_backdoor import (
    chamfer_vector,
    checkpoint_categories,
    fixed_target_set_cd,
    nearest_neighbour_calibration,
    paired_bootstrap,
    sample_group,
    summarize,
)
from models.vae_gaussian import GaussianVAE
from tools.pcd_backdoor_framework import (
    encoder_state_sha256,
    make_latent_mask_baseline,
    match_l2_with_linf,
    optimize_universal_latent_trigger,
)
from tools.pointcloud_normalization import load_pointcloud_target, tensor_sha256, validate_checkpoint_scale_mode
from utils.dataset import ShapeNetCore
from utils.misc import seed_all

METRIC_VERSION = "geometry-mask-v2-baseline-decoding-1.0"


def parse_args():
    p = argparse.ArgumentParser(description="Clean-checkpoint A/B target decodability audit")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--target_file", required=True)
    p.add_argument("--target_already_normalized", action="store_true")
    p.add_argument("--source_category", default="chair")
    p.add_argument("--target_category", default="airplane")
    p.add_argument("--scale_mode", default="shape_unit", choices=["shape_unit", "shape_bbox"])
    p.add_argument("--label", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pgd_steps", type=int, default=100)
    p.add_argument("--pgd_batch_size", type=int, default=8)
    p.add_argument("--pgd_lr", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--global_mask_path", default="")
    p.add_argument("--mask_mode", choices=["geometry_topk", "random_topk", "full_latent"], default="full_latent")
    p.add_argument("--active_ratio", type=float, default=0.25)
    p.add_argument("--match_trigger_l2_path", default="")
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--flexibility", type=float, default=0.0)
    p.add_argument("--calibration_samples", type=int, default=64)
    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument("--out_root", default="results_geometry_mask_v2/baseline_decodability")
    return p.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.out_root) / f"{args.label}_s{args.seed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    target, target_metadata = load_pointcloud_target(
        args.target_file, normalize=True,
        already_normalized=args.target_already_normalized,
        mode=args.scale_mode,
    )
    target = target.to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    validate_checkpoint_scale_mode(checkpoint["args"], args.scale_mode, "clean checkpoint")
    trained_categories = checkpoint_categories(checkpoint["args"])
    if trained_categories != [args.source_category]:
        raise RuntimeError(
            "Baseline audit requires the source-only checkpoint category to match "
            f"exactly: checkpoint={trained_categories}, source={args.source_category!r}"
        )
    model = GaussianVAE(checkpoint["args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    train = ShapeNetCore(
        args.dataset_path, [args.source_category], "train", args.scale_mode
    )
    test = ShapeNetCore(
        args.dataset_path, [args.source_category], "test", args.scale_mode
    )
    target_train = ShapeNetCore(
        args.dataset_path, [args.target_category], "train", args.scale_mode
    )
    generator = torch.Generator().manual_seed(args.seed + 101)
    pgd_loader = DataLoader(train, batch_size=args.pgd_batch_size, shuffle=True,
                            num_workers=0, generator=generator)
    if args.global_mask_path:
        global_mask_path = Path(args.global_mask_path)
        global_mask = torch.load(global_mask_path, map_location=device)
        if global_mask.shape != (1, model.encoder.zdim):
            raise ValueError("Global mask latent dimension mismatch")
        metadata_path = global_mask_path.parent / "mask_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError("Global mask has no sibling mask_metadata.json")
        mask_metadata = json.loads(metadata_path.read_text())
        if mask_metadata.get("encoder_sha256") != encoder_state_sha256(model.encoder):
            raise RuntimeError("Global mask encoder hash mismatch")
        if mask_metadata.get("scale_mode") != args.scale_mode:
            raise RuntimeError("Global mask normalization mode mismatch")
    elif args.mask_mode == "full_latent":
        global_mask = torch.ones(1, model.encoder.zdim, device=device)
    else:
        raise ValueError("--global_mask_path is required for masked trigger modes")
    effective_mask = make_latent_mask_baseline(
        global_mask, args.mask_mode, args.active_ratio, args.seed
    )
    trigger, pgd_log = optimize_universal_latent_trigger(
        model, pgd_loader, target, effective_mask,
        steps=args.pgd_steps, lr=args.pgd_lr, eps=args.eps,
        seed=args.seed + 303,
    )
    if args.match_trigger_l2_path:
        reference = torch.load(args.match_trigger_l2_path, map_location=device)
        trigger = match_l2_with_linf(
            trigger, float(reference.norm()), args.eps, support=(effective_mask != 0)
        )
    torch.save(trigger.cpu(), out_dir / "trigger.pt")
    torch.save(effective_mask.cpu(), out_dir / "effective_mask.pt")
    (out_dir / "pgd_log.json").write_text(json.dumps(pgd_log, indent=2))

    threshold, calibration = nearest_neighbour_calibration(
        target_train, device, args.calibration_samples, chunk_size=4
    )
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    values = {group: {metric: [] for metric in ("target_cd", "source_cd")} for group in "ABT"}
    samples = {group: [] for group in "ABT"}
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 9000)
    generated = 0
    with torch.no_grad():
        target_z, _ = model.encoder(target)
        for batch_index, batch in enumerate(loader):
            if generated >= args.num_samples:
                break
            source = batch["pointcloud"].to(device)
            source = source[: min(source.size(0), args.num_samples - generated)]
            latent, _ = model.encoder(source)
            initial_noise = torch.randn(
                source.size(0), args.num_points, 3,
                generator=noise_generator, device=device,
            )
            contexts = {
                "A": latent,
                "B": latent + trigger.expand(source.size(0), -1),
                "T": target_z.expand(source.size(0), -1),
            }
            reverse_seed = args.seed + 100000 + batch_index
            for group, context in contexts.items():
                output = sample_group(model, context, initial_noise.clone(), reverse_seed, args)
                samples[group].append(output.cpu())
                target_batch = target.expand(source.size(0), -1, -1)
                values[group]["target_cd"].extend(chamfer_vector(output, target_batch).cpu().tolist())
                values[group]["source_cd"].extend(chamfer_vector(output, source).cpu().tolist())
            generated += source.size(0)

    for group in "ABT":
        np.save(out_dir / f"samples_{group}.npy", torch.cat(samples[group]).numpy())
    a = np.asarray(values["A"]["target_cd"])
    b = np.asarray(values["B"]["target_cd"])
    gain = a - b
    metrics = {
        "metric_version": METRIC_VERSION,
        "purpose": "baseline selection only; not a geometry-mask result",
        "args": vars(args),
        "num_samples": generated,
        "calibrated_asr_threshold": threshold,
        "calibration_nearest_cd": summarize(calibration),
        "groups": {},
        "pretraining_trigger_gain_A_to_B": summarize(gain),
        "pretraining_trigger_gain_bootstrap_95ci": paired_bootstrap(
            gain, args.bootstrap_samples, args.seed + 7000
        ),
        "trigger": {
            "tensor_sha256": tensor_sha256(trigger),
            "l0": int((trigger != 0).sum()),
            "l2": float(trigger.norm()),
            "linf": float(trigger.abs().max()),
            "pgd_initial_loss": None if not pgd_log else pgd_log[0]["loss"],
            "pgd_final_loss": None if not pgd_log else pgd_log[-1]["loss"],
            "max_logged_grad_norm": max((x["grad_norm"] for x in pgd_log), default=0.0),
            "mask_outside_max": float((trigger * (effective_mask == 0)).abs().max()),
            "projection_violation": float((trigger.abs() - args.eps).clamp_min(0).max()),
            "universal_parameter_shape": list(trigger.shape),
        },
        "assets": {
            "checkpoint": {"path": str(Path(args.ckpt).resolve()), "sha256": file_sha256(args.ckpt)},
            "dataset": {"path": str(Path(args.dataset_path).resolve()), "sha256": file_sha256(args.dataset_path)},
            "target": target_metadata,
        },
        "out_dir": str(out_dir.resolve()),
    }
    for group in "ABT":
        target_cd = np.asarray(values[group]["target_cd"])
        metrics["groups"][group] = {
            "target_cd": summarize(target_cd),
            "fixed_target_set_cd": fixed_target_set_cd(target_cd),
            "source_cd": summarize(values[group]["source_cd"]),
            "calibrated_target_asr": float((target_cd <= threshold).mean()),
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"status": "complete", "metrics": str(out_dir / "metrics.json")}))


if __name__ == "__main__":
    main()
