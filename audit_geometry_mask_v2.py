import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_geometry_mask_backdoor import (
    chamfer_vector,
    checkpoint_categories,
    local_geometry_metrics,
    paired_bootstrap,
    summarize,
)
from models.vae_gaussian import GaussianVAE
from tools.pointcloud_normalization import validate_checkpoint_scale_mode
from tools.pcd_backdoor_framework import (
    build_global_latent_mask,
    encoder_state_sha256,
    make_latent_mask_baseline,
)
from utils.dataset import ShapeNetCore
from utils.misc import seed_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal sanity audit for a geometry latent mask"
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--global_mask_path", default="")
    parser.add_argument("--source_category", default="chair")
    parser.add_argument("--scale_mode", default="shape_unit", choices=["shape_unit", "shape_bbox"])
    parser.add_argument("--num_mask_references", type=int, default=64)
    parser.add_argument("--reference_batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--active_ratio", type=float, default=0.25)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--knn_k", type=int, default=15)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--flexibility", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out_dir", default="results_geometry_mask_v2/mask_sanity"
    )
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GaussianVAE(checkpoint["args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def rademacher(shape, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.randint(0, 2, shape, generator=generator, device=device)
    return values.to(torch.float32).mul_(2).sub_(1)


def main():
    args = parse_args()
    if not 0 < args.active_ratio <= 1:
        raise ValueError("active_ratio must be in (0,1]")
    if args.eps <= 0:
        raise ValueError("eps must be positive")
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.out_dir) / f"seed{args.seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    train_dataset = ShapeNetCore(
        args.dataset_path, [args.source_category], "train", args.scale_mode
    )
    test_dataset = ShapeNetCore(
        args.dataset_path, [args.source_category], "test", args.scale_mode
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    trained_categories = checkpoint_categories(checkpoint["args"])
    if trained_categories != [args.source_category]:
        raise RuntimeError(
            "Mask sanity source must match the source-only checkpoint exactly: "
            f"checkpoint={trained_categories}, source={args.source_category!r}"
        )
    model = load_model(args.ckpt, device)
    validate_checkpoint_scale_mode(
        checkpoint["args"], args.scale_mode,
        "clean checkpoint",
    )
    if args.global_mask_path:
        reused_path = Path(args.global_mask_path)
        global_mask = torch.load(reused_path, map_location="cpu")
        source_metadata_path = reused_path.parent / "mask_metadata.json"
        if not source_metadata_path.exists():
            raise RuntimeError("Reused sanity mask has no sibling mask_metadata.json")
        with open(source_metadata_path) as handle:
            mask_metadata = json.load(handle)
        model_hash = encoder_state_sha256(model.encoder)
        if mask_metadata.get("encoder_sha256") != model_hash:
            raise RuntimeError("Sanity mask encoder hash does not match the checkpoint")
        if mask_metadata.get("scale_mode") != args.scale_mode:
            raise RuntimeError("Sanity mask normalization mode does not match this run")
        if global_mask.shape != (1, model.encoder.zdim):
            raise ValueError("Sanity mask latent dimension does not match the encoder")
        mask_metadata = dict(mask_metadata)
        mask_metadata["reused_from"] = str(reused_path.resolve())
        per_shape_path = reused_path.parent / "per_shape_latent_masks.pt"
        if not per_shape_path.exists():
            raise RuntimeError("Reused sanity mask has no per_shape_latent_masks.pt")
        per_shape = torch.load(per_shape_path, map_location="cpu")
    else:
        reference_loader = make_loader(
            train_dataset, args.reference_batch_size, False, args.seed
        )
        global_mask, per_shape, examples, mask_metadata = build_global_latent_mask(
            model.encoder,
            reference_loader,
            num_reference_shapes=args.num_mask_references,
            knn_k=args.knn_k,
            device=device,
        )
        mask_metadata["scale_mode"] = args.scale_mode
        torch.save(per_shape, out_dir / "per_shape_latent_masks.pt")
        arrays = {}
        for index, example in enumerate(examples):
            for key, value in example.items():
                arrays[f"{key}_{index}"] = value.squeeze(0).numpy()
        np.savez(out_dir / "point_mask_examples.npz", **arrays)
    torch.save(global_mask, out_dir / "global_mask.pt")
    with open(out_dir / "mask_metadata.json", "w") as handle:
        json.dump(mask_metadata, handle, indent=2)

    mask_modes = {
        "high_mask": "geometry_topk",
        "low_mask": "inverse_geometry",
        "random_mask": "random_topk",
    }
    masks = {
        name: make_latent_mask_baseline(
            global_mask.to(device), mode, args.active_ratio, args.seed + 17
        )
        for name, mode in mask_modes.items()
    }
    direction = rademacher(global_mask.shape, args.seed + 23, device)
    deltas = {
        name: direction * mask * args.eps for name, mask in masks.items()
    }
    budget = {
        name: {
            "l0": int((delta != 0).sum()),
            "l2": float(delta.norm()),
            "linf": float(delta.abs().max()),
        }
        for name, delta in deltas.items()
    }

    loader = make_loader(test_dataset, args.batch_size, False, args.seed + 1)
    metric_names = (
        "global_cd",
        "edge_cd",
        "normal_consistency",
        "curvature_change",
        "decoder_displacement",
    )
    values = {
        name: {key: [] for key in metric_names} for name in masks
    }
    sample_outputs = {name: [] for name in ("clean", *masks.keys())}
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 101)
    generated = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if generated >= args.num_samples:
                break
            source = batch["pointcloud"].to(device)
            source = source[: min(source.size(0), args.num_samples - generated)]
            latent, _ = model.encoder(source)
            initial_noise = torch.randn(
                source.size(0),
                args.num_points,
                3,
                generator=noise_generator,
                device=device,
            )
            reverse_seed = args.seed + 1000 + batch_index
            torch.manual_seed(reverse_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(reverse_seed)
            clean = model.sample(
                latent,
                args.num_points,
                args.flexibility,
                initial_x_T=initial_noise.clone(),
            )
            sample_outputs["clean"].append(clean.cpu())
            for name, delta in deltas.items():
                torch.manual_seed(reverse_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(reverse_seed)
                perturbed = model.sample(
                    latent + delta.expand(source.size(0), -1),
                    args.num_points,
                    args.flexibility,
                    initial_x_T=initial_noise.clone(),
                )
                global_cd = chamfer_vector(clean, perturbed)
                edge, normal, curvature, displacement = local_geometry_metrics(
                    clean, perturbed, knn_k=args.knn_k
                )
                for key, tensor in (
                    ("global_cd", global_cd),
                    ("edge_cd", edge),
                    ("normal_consistency", normal),
                    ("curvature_change", curvature),
                    ("decoder_displacement", displacement),
                ):
                    values[name][key].extend(tensor.cpu().tolist())
                sample_outputs[name].append(perturbed.cpu())
            generated += source.size(0)

    summaries = {
        name: {
            metric: summarize(metric_values)
            for metric, metric_values in group.items()
        }
        for name, group in values.items()
    }
    paired_differences = {}
    for control in ("random_mask", "low_mask"):
        comparison = f"high_mask_minus_{control}"
        paired_differences[comparison] = {}
        for metric in metric_names:
            difference = np.asarray(values["high_mask"][metric]) - np.asarray(values[control][metric])
            paired_differences[comparison][metric] = {
                "summary": summarize(difference),
                "bootstrap_95ci": paired_bootstrap(
                    difference, 10000, args.seed + 5000 + len(paired_differences) * 101
                ),
                "preferred_direction": "positive" if metric == "normal_consistency" else "negative",
            }

    if per_shape.dim() != 2 or per_shape.size(1) != global_mask.size(1):
        raise ValueError("Per-shape latent masks are incompatible with the global mask")
    active_count = int(masks["high_mask"].sum())
    per_shape_topk = torch.zeros_like(per_shape, dtype=torch.bool)
    per_shape_indices = per_shape.topk(active_count, dim=1).indices
    per_shape_topk.scatter_(1, per_shape_indices, True)
    global_support = masks["high_mask"].cpu().bool().squeeze(0)
    support_frequency = per_shape_topk.float().mean(dim=0)
    intersections = (per_shape_topk[:, None, :] & per_shape_topk[None, :, :]).sum(dim=2).float()
    unions = (per_shape_topk[:, None, :] | per_shape_topk[None, :, :]).sum(dim=2).clamp_min(1).float()
    upper = torch.triu_indices(per_shape.size(0), per_shape.size(0), offset=1)
    pairwise_jaccard = (intersections / unions)[upper[0], upper[1]]
    variance_by_dim = per_shape.to(torch.float64).var(dim=0, unbiased=False)
    mask_stability = {
        "num_reference_shapes": int(per_shape.size(0)),
        "per_shape_variance_by_dim": summarize(variance_by_dim.numpy()),
        "global_active_dimension_frequency": summarize(support_frequency[global_support].numpy()),
        "global_inactive_dimension_frequency": summarize(support_frequency[~global_support].numpy()),
        "pairwise_topk_jaccard": summarize(pairwise_jaccard.numpy()),
    }
    random_edge_ci = paired_differences["high_mask_minus_random_mask"]["edge_cd"]["bootstrap_95ci"]
    low_edge_ci = paired_differences["high_mask_minus_low_mask"]["edge_cd"]["bootstrap_95ci"]
    go = random_edge_ci[1] < 0 and low_edge_ci[1] < 0
    report = {
        "args": vars(args),
        "num_samples": generated,
        "mask_metadata": mask_metadata,
        "budget": budget,
        "metrics": summaries,
        "paired_differences": paired_differences,
        "mask_stability": mask_stability,
        "go_no_go": {
            "pass": bool(go),
            "criterion": (
                "paired high-minus-random and high-minus-low edge-CD bootstrap "
                "95% CI upper bounds are both < 0"
            ),
            "claim_if_failed": (
                "Do not use geometry-aware terminology before revising the mask."
            ),
        },
    }
    with open(out_dir / "mask_sanity.json", "w") as handle:
        json.dump(report, handle, indent=2)
    for name, chunks in sample_outputs.items():
        np.save(out_dir / f"samples_{name}.npy", torch.cat(chunks).numpy())
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(out_dir / "mask_sanity.json"),
                "go": go,
            }
        )
    )


if __name__ == "__main__":
    main()
