import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader

from models.vae_gaussian import GaussianVAE
from tools.pcd_backdoor_framework import compute_geometric_mask_v2
from tools.pointcloud_normalization import load_pointcloud_target, validate_checkpoint_scale_mode
from utils.dataset import ShapeNetCore
from utils.misc import seed_all


METRIC_VERSION = "geometry-mask-v2.1-fixed-target-set-cd"


def parse_args():
    parser = argparse.ArgumentParser(description="Paired A/B/C/D evaluation for geometry-mask v2")
    parser.add_argument("--clean_ckpt", required=True)
    parser.add_argument("--bd_ckpt", required=True)
    parser.add_argument("--trigger_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--target_already_normalized", action="store_true")
    parser.add_argument("--source_category", default="chair")
    parser.add_argument("--target_category", default="airplane")
    parser.add_argument("--scale_mode", default="shape_unit", choices=["shape_unit", "shape_bbox"])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flexibility", type=float, default=0.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--calibration_chunk_size", type=int, default=4)
    parser.add_argument("--classifier_path", default="")
    parser.add_argument("--target_class_index", type=int, default=0)
    parser.add_argument("--out_dir", default="results_geometry_mask_v2/evaluation")
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GaussianVAE(checkpoint["args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def chamfer_vector(x, y):
    distances = torch.cdist(x, y)
    return distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1)


def paired_bootstrap(values, num_bootstrap, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(num_bootstrap, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def fixed_target_set_cd(values):
    """Set metrics for a singleton (Dirac) fixed-target distribution.

    ``average_cd`` averages all generated-to-fixed-target distances. Under the
    standard reference-to-sample MMD-CD definition, a singleton reference set
    reduces exactly to the minimum of those distances. Keeping both prevents
    an optimistic minimum from being mislabeled as an average attack result.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("fixed-target CD values must be a non-empty finite vector")
    return {
        "definition": (
            "average_cd=mean_s CD(s,target); mmd_cd=mean_r min_s CD(s,r), "
            "which for the singleton fixed target is min_s CD(s,target)"
        ),
        "target_distribution": "singleton_fixed_target",
        "average_cd": float(values.mean()),
        "median_cd": float(np.median(values)),
        "mmd_cd": float(values.min()),
        "num_generated": int(values.size),
        "num_target_references": 1,
    }


def checkpoint_categories(checkpoint_args):
    categories = (
        checkpoint_args.get("categories")
        if isinstance(checkpoint_args, dict)
        else getattr(checkpoint_args, "categories", None)
    )
    if isinstance(categories, str):
        categories = [item for item in categories.split(",") if item]
    return [] if categories is None else list(categories)


def nearest_neighbour_calibration(dataset, device, count, chunk_size=4):
    """Calibrate fixed-target ASR without materializing all pairwise point distances."""
    count = min(count, len(dataset))
    if count < 2:
        raise ValueError("At least two target-training shapes are required for ASR calibration")
    if chunk_size < 1:
        raise ValueError("calibration chunk_size must be positive")
    clouds = [dataset[index]["pointcloud"].cpu() for index in range(count)]
    nearest = []
    with torch.no_grad():
        for index in range(count):
            best = float("inf")
            query = clouds[index].unsqueeze(0).to(device)
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                candidate_indices = [j for j in range(start, stop) if j != index]
                if not candidate_indices:
                    continue
                candidates = torch.stack([clouds[j] for j in candidate_indices]).to(device)
                repeated_query = query.expand(candidates.size(0), -1, -1)
                best = min(best, float(chamfer_vector(repeated_query, candidates).min()))
            nearest.append(best)
    values = np.asarray(nearest, dtype=np.float64)
    return float(np.quantile(values, 0.95)), values


def local_geometry_metrics(reference, perturbed, knn_k=15, edge_fraction=0.25):
    normals_ref, curvature_ref, _ = compute_geometric_mask_v2(reference, knn_k)
    normals_other, curvature_other, _ = compute_geometric_mask_v2(perturbed, knn_k)
    edge_count = max(1, int(reference.size(1) * edge_fraction))
    edge_indices = curvature_ref.topk(edge_count, dim=1).indices
    edge_ref = torch.gather(reference, 1, edge_indices.unsqueeze(-1).expand(-1, -1, 3))
    edge_other = torch.gather(perturbed, 1, edge_indices.unsqueeze(-1).expand(-1, -1, 3))
    edge_cd = chamfer_vector(edge_ref, edge_other)
    matched = torch.cdist(reference, perturbed).argmin(dim=2)
    matched_normals = torch.gather(normals_other, 1, matched.unsqueeze(-1).expand(-1, -1, 3))
    normal_consistency = (normals_ref * matched_normals).sum(dim=2).abs().mean(dim=1)
    curvature_change = (curvature_ref - torch.gather(curvature_other, 1, matched)).abs().mean(dim=1)
    displacement = torch.linalg.vector_norm(reference - perturbed, dim=2).mean(dim=1)
    return edge_cd, normal_consistency, curvature_change, displacement


def sample_group(model, context, initial_noise, seed, args):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return model.sample(
        context,
        args.num_points,
        flexibility=args.flexibility,
        initial_x_T=initial_noise,
    )


def semantic_predictions(classifier, clouds, target_index):
    if classifier is None:
        return None
    logits = classifier(clouds)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return (logits.argmax(dim=1) == target_index).float().cpu().numpy()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    target, target_metadata = load_pointcloud_target(
        args.target_file, normalize=True, already_normalized=args.target_already_normalized,
        mode=args.scale_mode,
    )
    target = target.to(device)
    clean_model = load_model(args.clean_ckpt, device)
    bd_model = load_model(args.bd_ckpt, device)
    clean_checkpoint = torch.load(args.clean_ckpt, map_location="cpu")
    bd_checkpoint = torch.load(args.bd_ckpt, map_location="cpu")
    validate_checkpoint_scale_mode(
        clean_checkpoint["args"], args.scale_mode,
        "clean checkpoint",
    )
    validate_checkpoint_scale_mode(
        bd_checkpoint["args"], args.scale_mode,
        "redirected checkpoint",
    )
    clean_categories = checkpoint_categories(clean_checkpoint["args"])
    redirected_categories = checkpoint_categories(bd_checkpoint["args"])
    if args.source_category not in clean_categories:
        raise RuntimeError(
            f"clean checkpoint categories={clean_categories} do not contain "
            f"source_category={args.source_category!r}"
        )
    if redirected_categories != clean_categories:
        raise RuntimeError(
            "Redirected checkpoint category provenance differs from its clean "
            f"baseline: clean={clean_categories}, redirected={redirected_categories}"
        )
    geometry_metadata = bd_checkpoint.get("geometry_v2")
    if not geometry_metadata or geometry_metadata.get("scale_mode") != args.scale_mode:
        raise RuntimeError(
            "Redirected checkpoint lacks matching geometry_v2.scale_mode provenance; "
            "legacy shape-bbox runs cannot be evaluated as shape-unit runs"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    trigger = torch.load(args.trigger_path, map_location=device)
    if trigger.dim() == 1:
        trigger = trigger.unsqueeze(0)
    if trigger.dim() != 2 or trigger.size(0) != 1:
        raise ValueError(f"Expected one universal trigger [1, D], got {tuple(trigger.shape)}")
    expected_dim = clean_model.encoder.zdim
    if trigger.size(1) != expected_dim:
        raise ValueError(f"Trigger dim {trigger.size(1)} != encoder dim {expected_dim}")

    from utils.dataset import cate_to_synsetid
    required = {args.source_category: "test", args.target_category: "train"}
    with h5py.File(args.dataset_path, "r") as dataset_file:
        missing = []
        for category, split in required.items():
            synset = cate_to_synsetid.get(category)
            if synset is None or synset not in dataset_file or split not in dataset_file[synset]:
                missing.append(f"{category}:{split}")
        if missing:
            raise ValueError(
                "Evaluator dataset is missing required category/split entries "
                f"{missing}: {args.dataset_path}"
            )
    try:
        source_dataset = ShapeNetCore(
            args.dataset_path, [args.source_category], "test", args.scale_mode
        )
        target_calibration_dataset = ShapeNetCore(
            args.dataset_path, [args.target_category], "train", args.scale_mode
        )
    except (KeyError, OSError) as error:
        raise ValueError(
            "The evaluator dataset must contain both source_category="
            f"{args.source_category!r} and target_category={args.target_category!r}: "
            f"{args.dataset_path}"
        ) from error
    threshold, calibration_values = nearest_neighbour_calibration(
        target_calibration_dataset,
        device,
        args.calibration_samples,
        chunk_size=args.calibration_chunk_size,
    )
    loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    classifier = None
    if args.classifier_path:
        classifier = torch.jit.load(args.classifier_path, map_location=device).eval()

    group_samples = {key: [] for key in "ABCD"}
    source_samples = []
    group_target_cd = {key: [] for key in "ABCD"}
    group_source_cd = {key: [] for key in "ABCD"}
    semantic = {key: [] for key in "ABCD"}
    geometry = {
        f"{metric}_{pair}": []
        for pair in ("AB", "CD")
        for metric in ("edge_cd", "normal", "curvature", "displacement")
    }
    generated = 0
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 9000)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if generated >= args.num_samples:
                break
            source = batch["pointcloud"].to(device)
            source = source[: min(source.size(0), args.num_samples - generated)]
            batch_size = source.size(0)
            clean_z, _ = clean_model.encoder(source)
            bd_z, _ = bd_model.encoder(source)
            initial_noise = torch.randn(
                batch_size, args.num_points, 3, generator=noise_generator, device=device
            )
            group_contexts = {
                "A": (clean_model, clean_z),
                "B": (clean_model, clean_z + trigger.expand(batch_size, -1)),
                "C": (bd_model, bd_z),
                "D": (bd_model, bd_z + trigger.expand(batch_size, -1)),
            }
            current = {}
            reverse_seed = args.seed + 100000 + batch_index
            for key, (model, context) in group_contexts.items():
                samples = sample_group(model, context, initial_noise.clone(), reverse_seed, args)
                current[key] = samples
                group_samples[key].append(samples.cpu())
                target_batch = target.expand(batch_size, -1, -1)
                group_target_cd[key].extend(chamfer_vector(samples, target_batch).cpu().tolist())
                group_source_cd[key].extend(chamfer_vector(samples, source).cpu().tolist())
                predictions = semantic_predictions(classifier, samples, args.target_class_index)
                if predictions is not None:
                    semantic[key].extend(predictions.tolist())
            for pair, left, right in (("AB", "A", "B"), ("CD", "C", "D")):
                edge, normals, curvature, displacement = local_geometry_metrics(
                    current[left], current[right]
                )
                geometry[f"edge_cd_{pair}"].extend(edge.cpu().tolist())
                geometry[f"normal_{pair}"].extend(normals.cpu().tolist())
                geometry[f"curvature_{pair}"].extend(curvature.cpu().tolist())
                geometry[f"displacement_{pair}"].extend(displacement.cpu().tolist())
            source_samples.append(source.cpu())
            generated += batch_size

    for key in "ABCD":
        np.save(out_dir / f"samples_{key}.npy", torch.cat(group_samples[key]).numpy())
    np.save(out_dir / "sources.npy", torch.cat(source_samples).numpy())
    np.savez(
        out_dir / "fixed_target_cd_per_sample.npz",
        **{key: np.asarray(group_target_cd[key], dtype=np.float64) for key in "ABCD"},
    )
    c_target = np.asarray(group_target_cd["C"])
    d_target = np.asarray(group_target_cd["D"])
    paired_gain = c_target - d_target
    metrics = {
        "metric_version": METRIC_VERSION,
        "num_samples": generated,
        "calibrated_asr_threshold": threshold,
        "calibration_nearest_cd": summarize(calibration_values),
        "target_metadata": target_metadata,
        "groups": {},
        "conditional_attack_gain": summarize(paired_gain),
        "conditional_attack_gain_bootstrap_95ci": paired_bootstrap(
            paired_gain, args.bootstrap_samples, args.seed + 7000
        ),
        "geometry_paired_A_B_and_C_D": {
            key: summarize(value) for key, value in geometry.items()
        },
        "trigger": {
            "l0": int((trigger != 0).sum()),
            "l2": float(trigger.norm()),
            "linf": float(trigger.abs().max()),
        },
    }
    for key in "ABCD":
        target_values = np.asarray(group_target_cd[key])
        metrics["groups"][key] = {
            "target_cd": summarize(target_values),
            "fixed_target_set_cd": fixed_target_set_cd(target_values),
            "source_cd": summarize(group_source_cd[key]),
            "calibrated_target_asr": float((target_values <= threshold).mean()),
            "semantic_asr": None if not semantic[key] else float(np.mean(semantic[key])),
        }
    metrics["clean_utility_relative_degradation_C_vs_A"] = (
        metrics["groups"]["C"]["source_cd"]["mean"]
        / max(metrics["groups"]["A"]["source_cd"]["mean"], 1e-12)
        - 1.0
    )
    metrics["target_leakage_gain_A_to_C"] = (
        metrics["groups"]["A"]["target_cd"]["mean"]
        - metrics["groups"]["C"]["target_cd"]["mean"]
    )
    metrics["pretraining_trigger_gain_A_to_B"] = (
        metrics["groups"]["A"]["target_cd"]["mean"]
        - metrics["groups"]["B"]["target_cd"]["mean"]
    )
    metrics["acceptance"] = {
        "conditional_gain_ci_excludes_zero": metrics["conditional_attack_gain_bootstrap_95ci"][0] > 0,
        "clean_utility_degradation_le_10pct": metrics["clean_utility_relative_degradation_C_vs_A"] <= 0.10,
    }
    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
