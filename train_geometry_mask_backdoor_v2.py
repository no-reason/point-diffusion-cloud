import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from models.common import gaussian_entropy, get_linear_scheduler, reparameterize_gaussian, standard_normal_logprob
from models.vae_gaussian import GaussianVAE
from tools.pcd_backdoor_framework import (
    bernoulli_poison_mask,
    build_global_latent_mask,
    encoder_state_sha256,
    make_latent_mask_baseline,
    optimize_universal_latent_trigger,
    project_mask_to_latent,
    compute_geometric_mask_v2,
    configure_encoder_policy,
    match_l2_with_linf,
)
from tools.pointcloud_normalization import (
    load_pointcloud_target,
    tensor_sha256,
    validate_checkpoint_scale_mode,
)
from utils.dataset import ShapeNetCore
from utils.misc import seed_all


MASK_MODES = (
    "geometry_soft",
    "geometry_topk",
    "random_topk",
    "full_latent",
    "inverse_geometry",
    "no_trigger",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Geometry-aware latent backdoor v2")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--target_already_normalized", action="store_true")
    parser.add_argument("--categories", default="chair")
    parser.add_argument(
        "--scale_mode", default="shape_unit", choices=["shape_unit", "shape_bbox"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_mode", choices=MASK_MODES, default="geometry_soft")
    parser.add_argument("--active_ratio", type=float, default=0.25)
    parser.add_argument("--num_mask_references", type=int, default=64)
    parser.add_argument("--mask_reference_batch_size", type=int, default=4)
    parser.add_argument("--knn_k", type=int, default=15)
    parser.add_argument("--reuse_global_mask", default="")
    parser.add_argument("--allow_unverified_mask", action="store_true")
    parser.add_argument("--pgd_steps", type=int, default=500)
    parser.add_argument("--pgd_batch_size", type=int, default=8)
    parser.add_argument("--pgd_lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--timesteps_per_step", type=int, default=5)
    parser.add_argument("--lambda_cd", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=1e-4)
    parser.add_argument("--match_trigger_l2_path", default="")
    parser.add_argument("--encoder_policy", choices=["frozen", "joint_fixed_mask"], default="frozen")
    parser.add_argument("--poison_rate", type=float, default=0.03125)
    parser.add_argument("--poison_loss_weight", type=float, default=8.0)
    parser.add_argument(
        "--poison_latent_mode",
        choices=["mean", "sample"],
        default="mean",
        help=(
            "Latent used by the poison branch. 'mean' matches trigger optimization "
            "and evaluation; 'sample' is retained as an explicit legacy ablation."
        ),
    )
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--end_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--drift_interval", type=int, default=500)
    parser.add_argument("--drift_recompute_mask", action="store_true")
    parser.add_argument("--log_root", default="logs_geometry_mask_v2")
    parser.add_argument("--results_root", default="results_geometry_mask_v2")
    parser.add_argument("--run_tag", default="geometry_mask_v2")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(run_dir):
    def run(*command):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    dirty_diff = run("git", "diff", "--no-ext-diff")
    diff_path = Path(run_dir) / "dirty.diff"
    diff_path.write_text(dirty_diff)
    source_paths = [
        "tools/pcd_backdoor_framework.py",
        "tools/pointcloud_normalization.py",
        "utils/dataset.py",
        "train_geometry_mask_backdoor_v2.py",
        "evaluate_geometry_mask_backdoor.py",
        "audit_geometry_mask_v2.py",
        "run_geometry_mask_v2_matrix.py",
    ]
    source_hashes = {
        path: file_sha256(path) for path in source_paths if Path(path).exists()
    }
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_status": run("git", "status", "--short"),
        "diff_stat": run("git", "diff", "--stat"),
        "dirty_diff_path": str(diff_path.resolve()),
        "dirty_diff_sha256": file_sha256(diff_path),
        "source_file_sha256": source_hashes,
    }


def dump_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2)


def append_jsonl(path, value):
    with open(path, "a") as handle:
        handle.write(json.dumps(value) + "\n")


def make_loader(dataset, batch_size, shuffle, seed, drop_last=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
        generator=generator,
    )


def save_mask_assets(run_dir, global_mask, per_shape, examples, metadata):
    torch.save(global_mask, run_dir / "global_mask.pt")
    torch.save(per_shape, run_dir / "per_shape_latent_masks.pt")
    dump_json(run_dir / "mask_metadata.json", metadata)
    arrays = {}
    for index, example in enumerate(examples):
        for key, value in example.items():
            arrays[f"{key}_{index}"] = value.squeeze(0).numpy()
    np.savez(run_dir / "point_mask_examples.npz", **arrays)


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())


def combine_sample_weighted_losses(
    loss_clean,
    loss_poison,
    clean_count,
    poison_count,
    batch_size,
    poison_loss_weight,
):
    """Combine subgroup means into a true per-sample Bernoulli objective."""
    if batch_size <= 0 or clean_count < 0 or poison_count < 0:
        raise ValueError("Invalid Bernoulli subgroup counts")
    if clean_count + poison_count != batch_size:
        raise ValueError("clean_count + poison_count must equal batch_size")
    return (
        loss_clean * (clean_count / batch_size)
        + poison_loss_weight * loss_poison * (poison_count / batch_size)
    )


def checkpoint_categories(checkpoint_args):
    categories = (
        checkpoint_args.get("categories")
        if isinstance(checkpoint_args, dict)
        else getattr(checkpoint_args, "categories", None)
    )
    if isinstance(categories, str):
        categories = [item for item in categories.split(",") if item]
    return [] if categories is None else list(categories)


def main():
    args = parse_args()
    if args.smoke:
        args.num_mask_references = min(args.num_mask_references, 1)
        args.pgd_steps = min(args.pgd_steps, 2)
        args.max_iters = min(args.max_iters, 2)
        args.save_interval = 1
        args.log_interval = 1
    if not 0.0 <= args.poison_rate <= 1.0:
        raise ValueError("poison_rate must be in [0,1]")
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    validate_checkpoint_scale_mode(checkpoint["args"], args.scale_mode, "clean checkpoint")
    requested_categories = args.categories.split(",")
    trained_categories = checkpoint_categories(checkpoint["args"])
    if requested_categories != trained_categories:
        raise RuntimeError(
            "Source categories must exactly match the clean checkpoint training "
            f"categories: requested={requested_categories}, checkpoint={trained_categories}"
        )

    config = vars(args).copy()
    config_hash = json_hash(config)[:12]
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_name = f"{args.run_tag}_{args.mask_mode}_{args.encoder_policy}_s{args.seed}_{config_hash}_{timestamp}"
    run_dir = Path(args.log_root) / run_name
    result_dir = Path(args.results_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    dump_json(run_dir / "args.json", config)

    target, target_metadata = load_pointcloud_target(
        args.target_file,
        normalize=True,
        save_normalized_to=run_dir / "target_normalized.npy",
        metadata_path=run_dir / "target_metadata.json",
        already_normalized=args.target_already_normalized,
        mode=args.scale_mode,
    )
    target = target.to(device)

    train_dataset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories.split(","),
        split="train",
        scale_mode=args.scale_mode,
    )
    reference_loader = make_loader(
        train_dataset, args.mask_reference_batch_size, False, args.seed, drop_last=False
    )
    pgd_loader = make_loader(
        train_dataset, args.pgd_batch_size, True, args.seed + 101, drop_last=True
    )
    train_loader = make_loader(
        train_dataset, args.train_batch_size, True, args.seed + 202, drop_last=True
    )

    model = GaussianVAE(checkpoint["args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    clean_encoder_hash = encoder_state_sha256(model.encoder)

    if args.reuse_global_mask:
        reused_mask_path = Path(args.reuse_global_mask)
        global_mask = torch.load(reused_mask_path, map_location="cpu")
        if not isinstance(global_mask, torch.Tensor):
            raise TypeError("Reused global mask must be a torch.Tensor")
        source_metadata_path = reused_mask_path.parent / "mask_metadata.json"
        if source_metadata_path.exists():
            with open(source_metadata_path) as handle:
                source_mask_metadata = json.load(handle)
            recorded_hash = source_mask_metadata.get("encoder_sha256")
            if not recorded_hash and not args.allow_unverified_mask:
                raise RuntimeError("Reused mask metadata has no encoder_sha256")
            if recorded_hash and recorded_hash != clean_encoder_hash:
                raise RuntimeError(
                    "Reused mask was built from a different encoder checkpoint"
                )
        elif not args.allow_unverified_mask:
            raise RuntimeError(
                "Reused mask has no sibling mask_metadata.json; pass "
                "--allow_unverified_mask only for a deliberate legacy ablation"
            )
        else:
            source_mask_metadata = {"verification": "explicitly bypassed"}
        expected_dim = model.encoder.zdim
        if global_mask.shape != (1, expected_dim):
            raise ValueError(
                f"Expected global mask [1,{expected_dim}], got {tuple(global_mask.shape)}"
            )
        if not torch.isfinite(global_mask).all():
            raise ValueError("Reused global mask contains non-finite values")
        recorded_dim = source_mask_metadata.get("latent_dim")
        if recorded_dim is not None and int(recorded_dim) != expected_dim:
            raise ValueError("Reused mask metadata latent_dim does not match the encoder")
        recorded_api = source_mask_metadata.get("api_version")
        if recorded_api is not None and int(recorded_api) != 2:
            raise ValueError(f"Unsupported reused mask api_version={recorded_api}")
        recorded_scale_mode = source_mask_metadata.get("scale_mode")
        if recorded_scale_mode != args.scale_mode:
            raise RuntimeError(
                "Reused mask normalization mismatch: "
                f"metadata={recorded_scale_mode!r}, run={args.scale_mode!r}"
            )
        per_shape, examples = torch.empty(0), []
        mask_metadata = dict(source_mask_metadata)
        mask_metadata.update({
            "reused_from": str(reused_mask_path.resolve()),
            "source_metadata": str(source_metadata_path.resolve()),
            "encoder_sha256": clean_encoder_hash,
        })
        torch.save(global_mask, run_dir / "global_mask.pt")
        torch.save(per_shape, run_dir / "per_shape_latent_masks.pt")
        dump_json(run_dir / "mask_metadata.json", mask_metadata)
    else:
        global_mask, per_shape, examples, mask_metadata = build_global_latent_mask(
            model.encoder,
            reference_loader,
            num_reference_shapes=args.num_mask_references,
            knn_k=args.knn_k,
            device=device,
        )
        mask_metadata["scale_mode"] = args.scale_mode
        save_mask_assets(run_dir, global_mask, per_shape, examples, mask_metadata)

    effective_mask = make_latent_mask_baseline(
        global_mask.to(device), args.mask_mode, args.active_ratio, args.seed
    )
    torch.save(effective_mask.cpu(), run_dir / "effective_mask.pt")

    if args.mask_mode == "no_trigger":
        trigger = torch.zeros_like(effective_mask)
        pgd_log = []
    else:
        trigger, pgd_log = optimize_universal_latent_trigger(
            model,
            pgd_loader,
            target,
            effective_mask,
            steps=args.pgd_steps,
            lr=args.pgd_lr,
            eps=args.eps,
            timesteps_per_step=args.timesteps_per_step,
            lambda_cd=args.lambda_cd,
            lambda_l2=args.lambda_l2,
            seed=args.seed + 303,
        )
    if args.match_trigger_l2_path:
        reference_trigger = torch.load(args.match_trigger_l2_path, map_location=device)
        reference_norm = float(reference_trigger.norm())
        trigger = match_l2_with_linf(
            trigger,
            target_l2=reference_norm,
            eps=args.eps,
            support=(effective_mask != 0),
        )
    trigger = trigger.detach()
    torch.save(trigger.cpu(), run_dir / "trigger.pt")
    dump_json(run_dir / "pgd_log.json", pgd_log)
    trigger_stats = {
        "l0": int((trigger != 0).sum()),
        "active_dimensions": int((trigger.abs().amax(dim=0) > 0).sum()),
        "l2": float(trigger.norm()),
        "linf": float(trigger.abs().max()),
        "mask_outside_max": float((trigger * (effective_mask == 0)).abs().max()),
        "sha256": tensor_sha256(trigger),
    }
    dump_json(run_dir / "trigger_metadata.json", trigger_stats)
    if trigger_stats["mask_outside_max"] >= 1e-7:
        raise RuntimeError("Trigger violates the masked-support acceptance criterion")

    trainable = configure_encoder_policy(model, args.encoder_policy)
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=max(args.max_iters // 2, 1),
        end_epoch=max(args.max_iters, 2),
        start_lr=args.lr,
        end_lr=args.end_lr,
    )

    audit_batch = next(iter(reference_loader))["pointcloud"][:1].to(device)
    with torch.no_grad():
        audit_latent_initial, _ = model.encoder(audit_batch)
        target_latent_initial, _ = model.encoder(target)
    _, _, audit_point_mask = compute_geometric_mask_v2(audit_batch, args.knn_k)
    audit_local_mask_initial = project_mask_to_latent(
        model.encoder, audit_batch, audit_point_mask
    ).detach()
    trigger_direction_initial = target_latent_initial - audit_latent_initial
    training_log = run_dir / "training.jsonl"
    drift_log = run_dir / "drift_audit.jsonl"
    cumulative_poison = 0
    cumulative_samples = 0
    train_iterator = iter(train_loader)
    poison_generator = torch.Generator(device=device).manual_seed(args.seed + 404)

    for iteration in range(1, args.max_iters + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        points = batch["pointcloud"].to(device)
        batch_size = points.size(0)
        poison_mask = bernoulli_poison_mask(
            batch_size, args.poison_rate, device=device, generator=poison_generator
        )
        clean_mask = ~poison_mask
        cumulative_poison += int(poison_mask.sum())
        cumulative_samples += batch_size
        optimizer.zero_grad(set_to_none=True)
        if args.encoder_policy == "frozen":
            model.diffusion.train()
            model.encoder.eval()
        else:
            model.train()

        loss_clean = torch.zeros((), device=device)
        loss_poison = torch.zeros((), device=device)
        if clean_mask.any():
            clean_points = points[clean_mask]
            clean_mu, clean_logvar = model.encoder(clean_points)
            clean_z = reparameterize_gaussian(clean_mu, clean_logvar)
            entropy = gaussian_entropy(logvar=clean_logvar)
            log_prior = standard_normal_logprob(clean_z).sum(dim=1)
            clean_kl = (-log_prior - entropy).mean()
            loss_clean = model.diffusion.get_loss(clean_points, clean_z) + args.kl_weight * clean_kl
        if poison_mask.any():
            poison_points = points[poison_mask]
            poison_mu, poison_logvar = model.encoder(poison_points)
            poison_z = (
                poison_mu
                if args.poison_latent_mode == "mean"
                else reparameterize_gaussian(poison_mu, poison_logvar)
            )
            poison_context = poison_z + trigger.expand(poison_points.size(0), -1)
            target_batch = target.expand(poison_points.size(0), -1, -1)
            loss_poison = model.diffusion.get_loss(target_batch, poison_context)
        clean_count = int(clean_mask.sum())
        poison_count = int(poison_mask.sum())
        loss = combine_sample_weighted_losses(
            loss_clean,
            loss_poison,
            clean_count,
            poison_count,
            batch_size,
            args.poison_loss_weight,
        )
        loss.backward()
        grad_norm = clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if iteration == 1 or iteration % args.log_interval == 0:
            record = {
                "iteration": iteration,
                "loss_total": float(loss.detach()),
                "loss_clean_raw": float(loss_clean.detach()),
                "loss_poison_raw": float(loss_poison.detach()),
                "loss_clean_weighted": float(
                    (loss_clean * (clean_count / batch_size)).detach()
                ),
                "loss_poison_weighted": float(
                    (
                        args.poison_loss_weight
                        * loss_poison
                        * (poison_count / batch_size)
                    ).detach()
                ),
                "poison_count": poison_count,
                "batch_size": batch_size,
                "nominal_poison_rate": args.poison_rate,
                "realized_poison_rate": cumulative_poison / cumulative_samples,
                "grad_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(training_log, record)
            print(json.dumps(record))

        if args.encoder_policy == "joint_fixed_mask" and iteration % args.drift_interval == 0:
            model.eval()
            with torch.no_grad():
                audit_latent, _ = model.encoder(audit_batch)
                target_latent, _ = model.encoder(target)
            current_direction = target_latent - audit_latent
            drift = {
                "iteration": iteration,
                "latent_mean_l2_drift": float((audit_latent - audit_latent_initial).norm()),
                "trigger_direction_cosine": cosine(current_direction, trigger_direction_initial),
                "source_target_distance": float(current_direction.norm()),
                "encoder_sha256": encoder_state_sha256(model.encoder),
            }
            if args.drift_recompute_mask:
                current_mask = project_mask_to_latent(
                    model.encoder, audit_batch, audit_point_mask
                )
                drift["audit_local_mask_cosine"] = cosine(
                    current_mask, audit_local_mask_initial
                )
            append_jsonl(drift_log, drift)
            if args.encoder_policy == "joint_fixed_mask":
                model.train()

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            checkpoint_path = run_dir / f"ckpt_{iteration}.pt"
            torch.save({
                "args": checkpoint["args"],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "geometry_v2": {
                    "iteration": iteration,
                    "mask_mode": args.mask_mode,
                    "encoder_policy": args.encoder_policy,
                    "scale_mode": args.scale_mode,
                    "training_objective_version": "geometry-mask-v2.2-sample-weighted-full-batch",
                    "poison_latent_mode": args.poison_latent_mode,
                    "nominal_poison_rate": args.poison_rate,
                    "realized_poison_rate": cumulative_poison / cumulative_samples,
                    "trigger_path": "trigger.pt",
                    "global_mask_path": "global_mask.pt",
                    "effective_mask_path": "effective_mask.pt",
                },
            }, checkpoint_path)

    final_encoder_hash = encoder_state_sha256(model.encoder)
    if args.encoder_policy == "frozen" and final_encoder_hash != clean_encoder_hash:
        raise RuntimeError("Frozen encoder changed during training")
    manifest = {
        "run_name": run_name,
        "config_hash": config_hash,
        "config": config,
        "paths": {"run_dir": str(run_dir.resolve()), "result_dir": str(result_dir.resolve())},
        "assets": {
            "checkpoint": {"path": os.path.abspath(args.ckpt), "sha256": file_sha256(args.ckpt)},
            "dataset": {"path": os.path.abspath(args.dataset_path), "sha256": file_sha256(args.dataset_path)},
            "target": target_metadata,
            "trigger": trigger_stats,
            "global_mask": {
                "path": str((run_dir / "global_mask.pt").resolve()),
                "sha256": file_sha256(run_dir / "global_mask.pt"),
                "tensor_sha256": tensor_sha256(global_mask),
            },
            "effective_mask": {
                "path": str((run_dir / "effective_mask.pt").resolve()),
                "sha256": file_sha256(run_dir / "effective_mask.pt"),
                "tensor_sha256": tensor_sha256(effective_mask),
            },
        },
        "encoder": {"initial_sha256": clean_encoder_hash, "final_sha256": final_encoder_hash},
        "poisoning": {"nominal_rate": args.poison_rate, "realized_rate": cumulative_poison / cumulative_samples, "poison_samples": cumulative_poison, "total_samples": cumulative_samples},
        "git": git_provenance(run_dir),
        "training_objective_version": "geometry-mask-v2.2-sample-weighted-full-batch",
        "metric_version": "geometry-mask-v2.0",
        "final_checkpoint": str((run_dir / f"ckpt_{args.max_iters}.pt").resolve()),
        "final_checkpoint_sha256": file_sha256(run_dir / f"ckpt_{args.max_iters}.pt"),
    }
    dump_json(run_dir / "manifest.json", manifest)
    dump_json(result_dir / "run_pointer.json", manifest)
    print(json.dumps({"status": "complete", "manifest": str(run_dir / "manifest.json")}))


if __name__ == "__main__":
    main()
