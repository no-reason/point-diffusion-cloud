import os
import sys
import subprocess
import time
import json

def run_cmd(cmd):
    print(f"[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERR] Command failed: {res.stderr}")
    else:
        print(f"[OK] Output: {res.stdout[:200]}...")
    return res.returncode, res.stdout, res.stderr

def main():
    PYTHON = "/root/anaconda3/envs/baddiffusion-img/bin/python"
    ROOT_DIR = "/data/personal_data/zyy/point-diffusion-cloud"
    os.chdir(ROOT_DIR)

    # 1. Clean Checkpoints & Target paths
    CLEAN_CHAIR_CKPT = "./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt"
    CLEAN_PLANE_CKPT = "./logs_gen/GEN_2026_07_09__16_01_08_Clean_Airplane_From_Scratch_KL001/ckpt_0.775660_300000.pt"

    TARGET_AIRPLANE = "./targets/stageC8E_fixed_airplane_target.npy"
    TARGET_CHAIR = "./targets/stage3_fixed_chair_target.npy"

    # Define Experiment Configurations
    experiments = [
        # --- Generalization Experiments ---
        {
            "tag": "Task_A_Chair_to_Airplane",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0,
        },
        {
            "tag": "Task_B_Chair_to_Chair",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_CHAIR,
            "target_name": "Chair",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0,
        },
        {
            "tag": "Task_C_Airplane_to_Chair",
            "cates": "airplane",
            "clean_ckpt": CLEAN_PLANE_CKPT,
            "target_file": TARGET_CHAIR,
            "target_name": "Chair",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0,
        },
        {
            "tag": "Task_D_Airplane_to_Airplane",
            "cates": "airplane",
            "clean_ckpt": CLEAN_PLANE_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0,
        },

        # --- Hyperparameter Ablations (on Chair -> Airplane) ---
        # Poison Rate Ablation
        {
            "tag": "Abl_PR0015_Chair_to_Airplane",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.015625, # 1.5625%
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0,
        },
        {
            "tag": "Abl_PR00625_Chair_to_Airplane",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.0625, # 6.25%
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0,
        },
        # PGD Epsilon Ablation
        {
            "tag": "Abl_Eps03_Chair_to_Airplane",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.3,
            "poison_loss_weight": 8.0,
        },
        {
            "tag": "Abl_Eps08_Chair_to_Airplane",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.8,
            "poison_loss_weight": 8.0,
        },
        # Poison Loss Weight Ablation
        {
            "tag": "Abl_Weight4_Chair_to_Airplane",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 4.0,
        },
        {
            "tag": "Abl_Weight12_Chair_to_Airplane",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_file": TARGET_AIRPLANE,
            "target_name": "Airplane",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 12.0,
        },
    ]

    summary_results = []

    print(f"=== Starting Generalization & Ablation Suite ({len(experiments)} Experiments) ===")

    for i, exp in enumerate(experiments):
        tag = exp["tag"]
        print(f"\n[{i+1}/{len(experiments)}] Running Experiment: {tag}")

        # Step 1: Train Model
        train_cmd = (
            f"{PYTHON} train_manifold_backdoor.py "
            f"--tag {tag} "
            f"--ckpt {exp['clean_ckpt']} "
            f"--target_file {exp['target_file']} "
            f"--dataset_path ./data/shapenet_v2pc15k_chair_airplane.h5 "
            f"--categories {exp['cates']} "
            f"--poison_rate {exp['poison_rate']} "
            f"--max_iters {exp['max_iters']} "
            f"--eps {exp['eps']} "
            f"--poison_loss_weight {exp['poison_loss_weight']} "
            f"--val_freq {exp['max_iters']}"
        )
        run_cmd(train_cmd)

        # Find saved log directory
        log_dirs = [d for d in os.listdir("./logs_stageC") if d.startswith(tag)]
        if not log_dirs:
            print(f"[ERR] Log dir for {tag} not found!")
            continue
        log_dir = os.path.join("./logs_stageC", sorted(log_dirs)[-1])
        bd_ckpt = os.path.join(log_dir, f"ckpt_{exp['max_iters']}.pt")
        delta_path = os.path.join(log_dir, "delta_masked.pt")

        # Step 2: Run Metrics Evaluation
        eval_json = os.path.join(log_dir, "eval_metrics.json")
        eval_cmd = (
            f"{PYTHON} evaluate_manifold_backdoor.py "
            f"--ckpt {bd_ckpt} "
            f"--delta_path {delta_path} "
            f"--target_file {exp['target_file']} "
            f"--dataset_path ./data/shapenet_v2pc15k_chair_airplane.h5 "
            f"--cates {exp['cates']} "
            f"--out_json {eval_json}"
        )
        run_cmd(eval_cmd)

        # Read evaluation metrics
        metrics = {}
        if os.path.exists(eval_json):
            with open(eval_json, "r") as f:
                metrics = json.load(f)

        # Step 3: Generate Plot Visualization
        out_png = os.path.join(log_dir, "visuals.png")
        plot_cmd = (
            f"{PYTHON} plot_manifold_visuals.py "
            f"--clean_ckpt {exp['clean_ckpt']} "
            f"--bd_ckpt {bd_ckpt} "
            f"--delta_path {delta_path} "
            f"--target_file {exp['target_file']} "
            f"--dataset_path ./data/shapenet_v2pc15k_chair_airplane.h5 "
            f"--cates {exp['cates']} "
            f"--target_name {exp['target_name']} "
            f"--out_path {out_png}"
        )
        run_cmd(plot_cmd)

        res_record = {
            "tag": tag,
            "config": exp,
            "log_dir": log_dir,
            "bd_ckpt": bd_ckpt,
            "visuals_png": out_png,
            "metrics": metrics,
        }
        summary_results.append(res_record)

        # Save incremental progress
        with open("./suite_summary.json", "w") as f:
            json.dump(summary_results, f, indent=4)

    print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
