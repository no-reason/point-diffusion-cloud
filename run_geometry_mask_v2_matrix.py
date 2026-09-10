import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fail-fast staged driver for geometry-mask v2 experiments"
    )
    parser.add_argument(
        "--phase", choices=["sanity", "main", "parameter", "encoder"], required=True
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--target_already_normalized", action="store_true")
    parser.add_argument("--global_mask_path", default="")
    parser.add_argument("--sanity_reports", nargs="*", default=[])
    parser.add_argument("--override_sanity_gate", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--active_ratio", type=float, default=0.25)
    parser.add_argument("--best_eps", type=float, default=0.2)
    parser.add_argument("--best_poison_rate", type=float, default=0.03125)
    parser.add_argument("--eps_grid", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    parser.add_argument(
        "--poison_rate_grid", type=float, nargs="+", default=[0.01, 0.03125, 0.05]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale_mode", default="shape_unit", choices=["shape_unit", "shape_bbox"])
    parser.add_argument("--num_eval_samples", type=int, default=128)
    parser.add_argument("--num_sanity_samples", type=int, default=64)
    parser.add_argument("--logs_root", default="logs_geometry_mask_v2")
    parser.add_argument("--results_root", default="results_geometry_mask_v2")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--include_extra_masks", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_streaming(command, dry_run=False):
    print(json.dumps({"command": command}))
    if dry_run:
        return None
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    final = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status") == "complete":
            final = value
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if final is None:
        raise RuntimeError(f"Command completed without a structured completion record: {command}")
    return final


def read_manifest(completion):
    with open(completion["manifest"]) as handle:
        return json.load(handle)


def check_sanity_gate(args):
    if args.phase == "sanity" or args.override_sanity_gate:
        return
    if len(args.sanity_reports) < 3:
        raise RuntimeError(
            "Provide three --sanity_reports (seeds/reference subsets 0,1,2), "
            "or explicitly use --override_sanity_gate."
        )
    passes = 0
    for path in args.sanity_reports:
        with open(path) as handle:
            report = json.load(handle)
        passes += int(report["go_no_go"]["pass"])
    if passes < 2:
        raise RuntimeError(
            f"Mask sanity gate failed: only {passes}/{len(args.sanity_reports)} passed"
        )


def common_train_command(args, seed, mode, eps, poison_rate, policy):
    command = [
        sys.executable,
        "train_geometry_mask_backdoor_v2.py",
        "--ckpt", args.ckpt,
        "--dataset_path", args.dataset_path,
        "--target_file", args.target_file,
        "--seed", str(seed),
        "--mask_mode", mode,
        "--active_ratio", str(args.active_ratio),
        "--eps", str(eps),
        "--poison_rate", str(poison_rate),
        "--encoder_policy", policy,
        "--device", args.device,
        "--scale_mode", args.scale_mode,
        "--log_root", args.logs_root,
        "--results_root", args.results_root,
    ]
    if args.target_already_normalized:
        command.append("--target_already_normalized")
    return command


def evaluate_run(args, manifest, seed):
    if args.skip_eval or args.dry_run:
        return None
    run_dir = Path(manifest["paths"]["run_dir"])
    evaluation_dir = Path(args.results_root) / "evaluation" / manifest["run_name"]
    command = [
        sys.executable,
        "evaluate_geometry_mask_backdoor.py",
        "--clean_ckpt", args.ckpt,
        "--bd_ckpt", manifest["final_checkpoint"],
        "--trigger_path", str(run_dir / "trigger.pt"),
        "--dataset_path", args.dataset_path,
        "--target_file", args.target_file,
        "--seed", str(seed),
        "--device", args.device,
        "--scale_mode", args.scale_mode,
        "--num_samples", str(args.num_eval_samples),
        "--out_dir", str(evaluation_dir),
    ]
    if args.target_already_normalized:
        command.append("--target_already_normalized")
    subprocess.run(command, cwd=ROOT, check=True)
    return str(evaluation_dir.resolve())


def execute_train(args, command, records):
    completion = run_streaming(command, args.dry_run)
    if completion is None:
        records.append({"command": command, "status": "dry_run"})
        return None
    manifest = read_manifest(completion)
    evaluation = evaluate_run(args, manifest, manifest["config"]["seed"])
    records.append(
        {
            "manifest": completion["manifest"],
            "evaluation": evaluation,
            "run_name": manifest["run_name"],
        }
    )
    return manifest


def run_sanity(args, records):
    shared_mask = args.global_mask_path
    for seed in args.seeds:
        command = [
            sys.executable,
            "audit_geometry_mask_v2.py",
            "--ckpt", args.ckpt,
            "--dataset_path", args.dataset_path,
            "--seed", str(seed),
            "--device", args.device,
            "--scale_mode", args.scale_mode,
            "--active_ratio", str(args.active_ratio),
            "--num_samples", str(args.num_sanity_samples),
            "--out_dir", str(Path(args.results_root) / "mask_sanity"),
        ]
        if shared_mask:
            command.extend(["--global_mask_path", shared_mask])
        completion = run_streaming(command, args.dry_run)
        records.append(completion or {"command": command, "status": "dry_run"})
        if completion is not None and not shared_mask:
            shared_mask = str(Path(completion["report"]).parent / "global_mask.pt")
        elif args.dry_run and not shared_mask:
            shared_mask = f"<global_mask_from_sanity_seed_{seed}>"


def run_main(args, records):
    shared_mask = args.global_mask_path
    for seed in args.seeds:
        geometry_command = common_train_command(
            args, seed, "geometry_topk", args.best_eps,
            args.best_poison_rate, "frozen"
        )
        if shared_mask:
            geometry_command.extend(["--reuse_global_mask", shared_mask])
        geometry_manifest = execute_train(args, geometry_command, records)
        if args.dry_run:
            if not shared_mask:
                shared_mask = f"<global_mask_from_geometry_seed_{seed}>"
            geometry_trigger = f"<trigger_from_geometry_seed_{seed}>"
        else:
            assert geometry_manifest is not None
            geometry_dir = Path(geometry_manifest["paths"]["run_dir"])
            shared_mask = str(geometry_dir / "global_mask.pt")
            geometry_trigger = str(geometry_dir / "trigger.pt")
        modes = ["random_topk", "full_latent", "no_trigger"]
        if args.include_extra_masks:
            modes.extend(["geometry_soft", "inverse_geometry"])
        for mode in modes:
            command = common_train_command(
                args, seed, mode, args.best_eps,
                args.best_poison_rate, "frozen"
            )
            command.extend(["--reuse_global_mask", shared_mask])
            if mode in ("random_topk", "full_latent"):
                command.extend(["--match_trigger_l2_path", geometry_trigger])
            execute_train(args, command, records)


def run_parameter(args, records):
    if not args.global_mask_path and not args.dry_run:
        raise RuntimeError("--global_mask_path is required for the parameter phase")
    for seed in args.seeds:
        for eps in args.eps_grid:
            for poison_rate in args.poison_rate_grid:
                command = common_train_command(
                    args, seed, "geometry_topk", eps, poison_rate, "frozen"
                )
                if args.global_mask_path:
                    command.extend(["--reuse_global_mask", args.global_mask_path])
                execute_train(args, command, records)


def run_encoder(args, records):
    if not args.global_mask_path and not args.dry_run:
        raise RuntimeError("--global_mask_path is required for the encoder phase")
    for seed in args.seeds:
        for policy in ("frozen", "joint_fixed_mask"):
            command = common_train_command(
                args, seed, "geometry_topk", args.best_eps,
                args.best_poison_rate, policy
            )
            if args.global_mask_path:
                command.extend(["--reuse_global_mask", args.global_mask_path])
            if policy == "joint_fixed_mask":
                command.append("--drift_recompute_mask")
            execute_train(args, command, records)


def main():
    args = parse_args()
    check_sanity_gate(args)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    matrix_dir = Path(args.results_root) / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    output = matrix_dir / f"{args.phase}_{timestamp}.json"
    records = []
    handlers = {
        "sanity": run_sanity,
        "main": run_main,
        "parameter": run_parameter,
        "encoder": run_encoder,
    }
    try:
        handlers[args.phase](args, records)
    except Exception:
        with open(output, "w") as handle:
            json.dump(
                {"status": "failed", "args": vars(args), "records": records},
                handle,
                indent=2,
            )
        raise
    with open(output, "w") as handle:
        json.dump(
            {"status": "complete", "args": vars(args), "records": records},
            handle,
            indent=2,
        )
    print(json.dumps({"status": "complete", "matrix_manifest": str(output)}))


if __name__ == "__main__":
    main()
