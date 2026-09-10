import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from models.vae_gaussian import GaussianVAE
from tools.pointcloud_normalization import load_pointcloud_target, validate_checkpoint_scale_mode
from utils.dataset import ShapeNetCore


def parse_args():
    p = argparse.ArgumentParser(description="Visualize formal geometry-mask-v2 results")
    p.add_argument("--clean_ckpt", required=True)
    p.add_argument("--bd_ckpt", required=True)
    p.add_argument("--trigger_path", required=True)
    p.add_argument("--effective_mask_path", required=True)
    p.add_argument("--evaluation_dir", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--target_file", required=True)
    p.add_argument("--source_category", default="chair")
    p.add_argument("--target_category", default="airplane")
    p.add_argument("--num_latent_per_class", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scale_mode", default="shape_unit", choices=["shape_unit", "shape_bbox"])
    p.add_argument("--out_dir", default="results_geometry_mask_v2/visualizations/formal_seed0_geometry")
    return p.parse_args()


def file_sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = GaussianVAE(ckpt["args"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def chamfer_to_fixed(clouds, target, device, chunk=4):
    values = []
    target = torch.from_numpy(target).float().unsqueeze(0).to(device)
    with torch.no_grad():
        for start in range(0, len(clouds), chunk):
            x = torch.from_numpy(clouds[start:start + chunk]).float().to(device)
            y = target.expand(x.size(0), -1, -1)
            dist = torch.cdist(x, y)
            cd = dist.amin(dim=2).mean(dim=1) + dist.amin(dim=1).mean(dim=1)
            values.extend(cd.cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def set_equal_3d(ax, clouds):
    points = np.concatenate(clouds, axis=0)
    shown = np.column_stack([points[:, 0], points[:, 2], points[:, 1]])
    center = (shown.min(axis=0) + shown.max(axis=0)) / 2
    radius = max((shown.max(axis=0) - shown.min(axis=0)).max() / 2, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Z", fontsize=8)
    ax.set_zlabel("Y (up)", fontsize=8)


def scatter_pc(ax, pc, color, title, elev, azim):
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1.0, c=color, alpha=0.82, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)


def encode_dataset(model, dataset, count, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= count:
                break
            x = batch["pointcloud"].to(device)
            x = x[: min(x.size(0), count - seen)]
            mu, _ = model.encoder(x)
            chunks.append(mu.cpu())
            seen += x.size(0)
    if seen != min(count, len(dataset)):
        raise RuntimeError(f"Encoded {seen}, expected {min(count, len(dataset))}")
    return torch.cat(chunks).numpy()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eval_dir = Path(args.evaluation_dir)
    with open(eval_dir / "metrics.json") as handle:
        evaluation_metrics = json.load(handle)
    evaluation_mode = evaluation_metrics.get("target_metadata", {}).get("normalization")
    if evaluation_mode != args.scale_mode:
        raise RuntimeError(
            f"Evaluation normalization={evaluation_mode!r} does not match "
            f"visualization scale_mode={args.scale_mode!r}"
        )
    clean_checkpoint = torch.load(args.clean_ckpt, map_location="cpu")
    bd_checkpoint = torch.load(args.bd_ckpt, map_location="cpu")
    validate_checkpoint_scale_mode(
        clean_checkpoint["args"], args.scale_mode, "clean checkpoint"
    )
    validate_checkpoint_scale_mode(
        bd_checkpoint["args"], args.scale_mode, "redirected checkpoint"
    )
    geometry_metadata = bd_checkpoint.get("geometry_v2")
    if not geometry_metadata or geometry_metadata.get("scale_mode") != args.scale_mode:
        raise RuntimeError("Redirected checkpoint normalization provenance mismatch")
    out.mkdir(parents=True, exist_ok=False)
    sources = np.load(eval_dir / "sources.npy")
    groups = {key: np.load(eval_dir / f"samples_{key}.npy") for key in "ABCD"}
    target_tensor, target_meta = load_pointcloud_target(
        args.target_file, normalize=True, mode=args.scale_mode
    )
    target = target_tensor.squeeze(0).numpy()

    d_target_cd = chamfer_to_fixed(groups["D"], target, device)
    median_cd = float(np.median(d_target_cd))
    selected = int(np.argmin(np.abs(d_target_cd - median_cd)))
    source, result = sources[selected], groups["D"][selected]

    views = [(22, -58), (12, 125)]
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    panels = [(source, "#277da1", f"Source {args.source_category} (input geometry)"),
              (target, "#f8961e", f"Fixed {args.target_category} target"),
              (result, "#43aa8b", "D: redirected result (geometry output)")]
    for row, (elev, azim) in enumerate(views):
        for col, (pc, color, title) in enumerate(panels):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            scatter_pc(ax, pc, color, title if row == 0 else f"view {row + 1}", elev, azim)
            set_equal_3d(ax, [source, target, result])
    fig.suptitle(
        f"Geometry-mask v2 formal seed0 — median-representative sample #{selected}\n"
        f"D→target unsquared Chamfer={d_target_cd[selected]:.4f}; trigger is added in 512-D latent space",
        fontsize=13,
    )
    fig.savefig(out / "source_target_result_3d.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    colors = {"A": "#577590", "B": "#f9844a", "C": "#90be6d", "D": "#f94144"}
    titles = {
        "A": "A: clean model + clean latent",
        "B": "B: clean model + latent trigger",
        "C": "C: redirected model + clean latent",
        "D": "D: redirected model + latent trigger",
    }
    for col, key in enumerate("ABCD"):
        ax = fig.add_subplot(2, 4, col + 1, projection="3d")
        scatter_pc(ax, groups[key][selected], colors[key], titles[key], 22, -58)
        set_equal_3d(ax, [groups[k][selected] for k in "ABCD"])
        ax = fig.add_subplot(2, 4, 4 + col + 1, projection="3d")
        scatter_pc(ax, groups[key][selected], colors[key], "alternate view", 12, 125)
        set_equal_3d(ax, [groups[k][selected] for k in "ABCD"])
    fig.suptitle(f"Paired A/B/C/D for the same source and reverse noise — sample #{selected}", fontsize=13)
    fig.savefig(out / "paired_ABCD_3d.png", dpi=220)
    plt.close(fig)

    clean = load_model(args.clean_ckpt, device)
    redirected = load_model(args.bd_ckpt, device)
    trigger = torch.load(args.trigger_path, map_location="cpu").float().numpy()
    effective_mask = torch.load(args.effective_mask_path, map_location="cpu").float().numpy()
    source_ds = ShapeNetCore(
        args.dataset_path, [args.source_category], "test", args.scale_mode
    )
    target_ds = ShapeNetCore(
        args.dataset_path, [args.target_category], "train", args.scale_mode
    )
    n = args.num_latent_per_class
    source_z = encode_dataset(clean, source_ds, n, device)
    source_z_redirected_encoder = encode_dataset(redirected, source_ds, n, device)
    target_class_z = encode_dataset(clean, target_ds, n, device)
    with torch.no_grad():
        target_z = clean.encoder(target_tensor.to(device))[0].cpu().numpy()
    source_triggered = source_z + trigger

    high_dim = np.concatenate([source_z, source_triggered, target_class_z, target_z], axis=0)
    pca_dim = min(50, high_dim.shape[1], high_dim.shape[0] - 1)
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    reduced = pca.fit_transform(high_dim)
    tsne = TSNE(
        n_components=2, perplexity=30, learning_rate="auto", init="pca",
        max_iter=1500, random_state=args.seed,
    )
    emb = tsne.fit_transform(reduced)
    slices = {
        f"{args.source_category} clean z": slice(0, n),
        f"{args.source_category} z + trigger": slice(n, 2 * n),
        f"{args.target_category} clean z": slice(2 * n, 3 * n),
        "fixed target z": slice(3 * n, 3 * n + 1),
    }
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    palette = {
        f"{args.source_category} clean z": "#277da1",
        f"{args.source_category} z + trigger": "#f94144",
        f"{args.target_category} clean z": "#f8961e", "fixed target z": "#111111",
    }
    markers = {
        f"{args.source_category} clean z": "o",
        f"{args.source_category} z + trigger": "^",
        f"{args.target_category} clean z": "s",
        "fixed target z": "*",
    }
    for label, sl in slices.items():
        size = 150 if label == "fixed target z" else 27
        ax.scatter(emb[sl, 0], emb[sl, 1], s=size, c=palette[label], marker=markers[label],
                   alpha=0.78, label=label, edgecolors="none")
    for i in range(min(32, n)):
        ax.plot([emb[i, 0], emb[n+i, 0]], [emb[i, 1], emb[n+i, 1]],
                color="#888888", alpha=0.18, linewidth=0.7)
    ax.set_title("t-SNE of 512-D encoder means and latent-triggered contexts\n"
                 f"Lines pair the same {args.source_category} before/after adding the universal latent trigger")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True)
    ax.grid(alpha=0.15)
    fig.savefig(out / "latent_tsne.png", dpi=240)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True, sharex=True)
    dims = np.arange(trigger.shape[1])
    ax1.bar(dims, trigger[0], width=1.0, color=np.where(trigger[0] >= 0, "#f94144", "#277da1"))
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_ylabel("trigger δ")
    ax1.set_title("512-D universal latent trigger (not a 3-D point perturbation)")
    ax2.fill_between(dims, 0, effective_mask[0], step="mid", color="#43aa8b")
    ax2.set_ylabel("binary mask")
    ax2.set_xlabel("latent dimension")
    ax2.set_ylim(-0.05, 1.05)
    fig.savefig(out / "latent_trigger_and_mask.png", dpi=220)
    plt.close(fig)

    np.savez(
        out / "latent_tsne_data.npz", source_z=source_z,
        source_triggered=source_triggered, target_class_z=target_class_z,
        target_z=target_z, tsne=emb,
        trigger=trigger, effective_mask=effective_mask,
    )
    metadata = {
        "selection_policy": "D target-CD closest to the sample median; fixed before plotting",
        "selected_index": selected,
        "selected_D_target_cd": float(d_target_cd[selected]),
        "median_D_target_cd": median_cd,
        "num_eval_samples": int(len(sources)),
        "num_latent_per_class": n,
        "source_category": args.source_category,
        "target_category": args.target_category,
        "latent_dim": int(source_z.shape[1]),
        "encoder_frozen_max_abs_latent_difference_clean_vs_redirected": float(np.max(np.abs(source_z - source_z_redirected_encoder))),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "tsne_warning": "t-SNE is a qualitative nonlinear projection; distances and cluster areas are not quantitative attack metrics.",
        "target_metadata": target_meta,
        "assets": {k: {"path": str(Path(v).resolve()), "sha256": file_sha256(v)} for k, v in {
            "clean_ckpt": args.clean_ckpt, "bd_ckpt": args.bd_ckpt,
            "trigger": args.trigger_path, "effective_mask": args.effective_mask_path,
            "sources": eval_dir / "sources.npy", "samples_D": eval_dir / "samples_D.npy",
        }.items()},
    }
    (out / "visualization_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"status": "complete", "out_dir": str(out), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
