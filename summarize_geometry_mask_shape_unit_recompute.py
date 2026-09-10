#!/usr/bin/env python3
"""Summarize the explicitly validated shape_unit recomputation matrix.

This intentionally rejects ambiguous matches instead of selecting a "latest" run.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("geometry_topk", "random_topk", "full_latent", "no_trigger")


def one(items, description):
    if len(items) != 1:
        raise RuntimeError(f"Expected exactly one {description}, found {len(items)}: {items}")
    return items[0]


def load_matrix(log_root: Path, evaluation_root: Path, tag: str, seeds):
    records = []
    manifests = list(log_root.glob("*/manifest.json"))
    for seed in seeds:
        for method in METHODS:
            candidates = []
            for path in manifests:
                data = json.loads(path.read_text())
                cfg = data["config"]
                if (
                    cfg.get("run_tag") == tag
                    and cfg.get("seed") == seed
                    and cfg.get("mask_mode") == method
                    and cfg.get("scale_mode") == "shape_unit"
                    and data.get("training_objective_version")
                    == "geometry-mask-v2.2-sample-weighted-full-batch"
                ):
                    candidates.append((path, data))
            manifest_path, manifest = one(candidates, f"{tag}/{method}/seed{seed}")
            metrics_path = evaluation_root / manifest["run_name"] / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            metrics = json.loads(metrics_path.read_text())
            if metrics["target_metadata"]["normalization"] != "shape_unit":
                raise RuntimeError(f"Non-shape_unit metrics: {metrics_path}")
            groups = metrics["groups"]
            records.append(
                {
                    "seed": seed,
                    "method": method,
                    "manifest": str(manifest_path.resolve()),
                    "metrics": str(metrics_path.resolve()),
                    "target_cd": {g: groups[g]["target_cd"]["mean"] for g in "ABCD"},
                    "source_cd": {g: groups[g]["source_cd"]["mean"] for g in "ABCD"},
                    "calibrated_asr": {g: groups[g]["calibrated_target_asr"] for g in "ABCD"},
                    "conditional_gain": metrics["conditional_attack_gain"]["mean"],
                    "conditional_gain_ci": metrics["conditional_attack_gain_bootstrap_95ci"],
                    "clean_utility_relative_degradation": metrics[
                        "clean_utility_relative_degradation_C_vs_A"
                    ],
                    "target_leakage_gain": metrics["target_leakage_gain_A_to_C"],
                    "pretraining_trigger_gain": metrics["pretraining_trigger_gain_A_to_B"],
                    "trigger": metrics["trigger"],
                    "acceptance": metrics["acceptance"],
                    "geometry_paired": metrics["geometry_paired_A_B_and_C_D"],
                }
            )
    return records


def aggregate(records):
    result = {}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        result[method] = {}
        for field in ("conditional_gain", "clean_utility_relative_degradation", "target_leakage_gain", "pretraining_trigger_gain"):
            values = np.asarray([r[field] for r in rows], dtype=np.float64)
            result[method][field] = {
                "mean": float(values.mean()),
                "std_across_seeds_ddof1": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        for metric in ("target_cd", "source_cd", "calibrated_asr"):
            result[method][metric] = {}
            for group in "ABCD":
                values = np.asarray([r[metric][group] for r in rows], dtype=np.float64)
                result[method][metric][group] = {
                    "mean": float(values.mean()),
                    "std_across_seeds_ddof1": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
    return result


def plot_summary(records, out_dir):
    summary = aggregate(records)
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in (
        (axes[0], "conditional_gain", "Conditional gain C-D (higher is better)"),
        (axes[1], "clean_utility_relative_degradation", "Clean source-CD degradation C vs A"),
        (axes[2], "target_leakage_gain", "Target leakage gain A-C"),
    ):
        means = [summary[m][key]["mean"] for m in METHODS]
        stds = [summary[m][key]["std_across_seeds_ddof1"] for m in METHODS]
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        if key == "clean_utility_relative_degradation":
            ax.axhline(0.1, color="red", linestyle="--", linewidth=1, label="10% limit")
            ax.legend()
        ax.set_xticks(x, METHODS, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Shape-unit recomputation: mean aggregation, active ratio 50%, 3 seeds")
    fig.tight_layout()
    fig.savefig(out_dir / "matched_controls_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.18
    for j, group in enumerate("ABCD"):
        axes[0].bar(x + (j - 1.5) * width, [summary[m]["target_cd"][group]["mean"] for m in METHODS], width, label=group)
        axes[1].bar(x + (j - 1.5) * width, [summary[m]["source_cd"][group]["mean"] for m in METHODS], width, label=group)
    for ax, title in zip(axes, ("Target CD (lower is better)", "Source CD (lower is better)")):
        ax.set_xticks(x, METHODS, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "abcd_target_source_cd.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=Path, default=Path("logs_geometry_mask_v2/shape_unit_recompute"))
    parser.add_argument("--evaluation_root", type=Path, default=Path("results_geometry_mask_v2/shape_unit_recompute/evaluation"))
    parser.add_argument("--out_dir", type=Path, default=Path("results_geometry_mask_v2/shape_unit_recompute/summary"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "geometry_mask_v2_shape_unit_mean_r0p5_diagnostic_v22"
    records = load_matrix(args.log_root, args.evaluation_root, tag, (0, 1, 2))
    payload = {
        "scope": "shape_unit recomputation; mean aggregation; active_ratio=0.5; frozen encoder",
        "selection_policy": "unique exact config match; never latest-directory sorting",
        "records": records,
        "method_seed_mean_std": aggregate(records),
    }
    output = args.out_dir / "formal_main_summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    plot_summary(records, args.out_dir)
    print(output)


if __name__ == "__main__":
    main()
