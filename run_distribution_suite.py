import os
import sys
import subprocess
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

    CLEAN_CHAIR_CKPT = "./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt"
    CLEAN_PLANE_CKPT = "./logs_gen/GEN_2026_07_09__16_01_08_Clean_Airplane_From_Scratch_KL001/ckpt_0.775660_300000.pt"
    DATASET_PATH = "./data/shapenet_v2pc15k_chair_airplane.h5"

    experiments = [
        # 1. Distribution Steering Generalization Tasks
        {
            "tag": "Dist_Task_A_Chair_to_Airplane_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0
        },
        {
            "tag": "Dist_Task_B_Chair_to_Chair_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "chair",
            "target_name": "Chair Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0
        },
        {
            "tag": "Dist_Task_C_Airplane_to_Chair_Dist",
            "cates": "airplane",
            "clean_ckpt": CLEAN_PLANE_CKPT,
            "target_mode": "distribution",
            "target_categories": "chair",
            "target_name": "Chair Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0
        },
        {
            "tag": "Dist_Task_D_Airplane_to_Airplane_Dist",
            "cates": "airplane",
            "clean_ckpt": CLEAN_PLANE_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0
        },
        # 2. Distribution Steering Hyperparameter Ablations
        {
            "tag": "Dist_Abl_PR0015_Chair_to_Airplane_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.015625,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0
        },
        {
            "tag": "Dist_Abl_PR00625_Chair_to_Airplane_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.0625,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 8.0
        },
        {
            "tag": "Dist_Abl_Eps03_Chair_to_Airplane_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.3,
            "poison_loss_weight": 8.0
        },
        {
            "tag": "Dist_Abl_Eps08_Chair_to_Airplane_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.8,
            "poison_loss_weight": 8.0
        },
        {
            "tag": "Dist_Abl_Weight4_Chair_to_Airplane_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 4.0
        },
        {
            "tag": "Dist_Abl_Weight12_Chair_to_Airplane_Dist",
            "cates": "chair",
            "clean_ckpt": CLEAN_CHAIR_CKPT,
            "target_mode": "distribution",
            "target_categories": "airplane",
            "target_name": "Airplane Distribution",
            "poison_rate": 0.03125,
            "max_iters": 10000,
            "eps": 0.5,
            "poison_loss_weight": 12.0
        }
    ]

    summary_file = "./distribution_suite_summary.json"
    summary_data = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                summary_data = json.load(f)
        except Exception:
            summary_data = []

    completed_tags = [item["tag"] for item in summary_data]

    for exp in experiments:
        tag = exp["tag"]
        if tag in completed_tags:
            print(f"\n>>> Skipping already completed experiment: {tag}")
            continue

        print(f"\n=======================================================")
        print(f">>> Starting Distribution Steering Experiment: {tag}")
        print(f"=======================================================")

        # Step 1: Train Poisoned Model
        train_cmd = (
            f"{PYTHON} train_manifold_backdoor.py "
            f"--tag {tag} "
            f"--ckpt {exp['clean_ckpt']} "
            f"--dataset_path {DATASET_PATH} "
            f"--categories {exp['cates']} "
            f"--target_mode {exp['target_mode']} "
            f"--target_categories {exp['target_categories']} "
            f"--poison_rate {exp['poison_rate']} "
            f"--max_iters {exp['max_iters']} "
            f"--eps {exp['eps']} "
            f"--poison_loss_weight {exp['poison_loss_weight']} "
            f"--val_freq 10000"
        )
        run_cmd(train_cmd)

        # Locate log dir
        log_dirs = [d for d in os.listdir("./logs_stageC") if d.startswith(tag)]
        if not log_dirs:
            print(f"[ERR] Log dir for {tag} not found! Skipping evaluation.")
            continue
        log_dir = os.path.join("./logs_stageC", sorted(log_dirs)[-1])
        bd_ckpt = os.path.join(log_dir, f"ckpt_{exp['max_iters']}.pt")
        delta_path = os.path.join(log_dir, "delta_masked.pt")

        # Step 2: Run Distribution Metrics Evaluation
        eval_json = os.path.join(log_dir, "eval_distribution_metrics.json")
        eval_cmd = (
            f"{PYTHON} evaluate_manifold_backdoor.py "
            f"--ckpt {bd_ckpt} "
            f"--delta_path {delta_path} "
            f"--dataset_path {DATASET_PATH} "
            f"--cates {exp['cates']} "
            f"--target_mode {exp['target_mode']} "
            f"--target_categories {exp['target_categories']} "
            f"--out_json {eval_json}"
        )
        run_cmd(eval_cmd)

        metrics = {}
        if os.path.exists(eval_json):
            with open(eval_json, "r") as f:
                metrics = json.load(f)

        out_png = os.path.join(log_dir, "visuals_distribution.png")
        plot_cmd = (
            f"{PYTHON} plot_manifold_visuals.py "
            f"--clean_ckpt {exp['clean_ckpt']} "
            f"--bd_ckpt {bd_ckpt} "
            f"--delta_path {delta_path} "
            f"--dataset_path {DATASET_PATH} "
            f"--cates {exp['cates']} "
            f"--target_name '{exp['target_name']}' "
            f"--target_mode {exp['target_mode']} "
            f"--target_categories {exp['target_categories']} "
            f"--out_path {out_png}"
        )
        run_cmd(plot_cmd)

        rec = {
            "tag": tag,
            "config": exp,
            "log_dir": log_dir,
            "bd_ckpt": bd_ckpt,
            "visuals_png": out_png,
            "metrics": metrics
        }
        summary_data.append(rec)

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)

        print(f"✅ Successfully finished and logged: {tag}")

    print("\n🎉 ALL DISTRIBUTION STEERING EXPERIMENTS COMPLETED!")

if __name__ == "__main__":
    main()
