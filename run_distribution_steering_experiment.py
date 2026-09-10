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
        print(f"[OK] Output: {res.stdout[:250]}...")
    return res.returncode, res.stdout, res.stderr

def main():
    PYTHON = "/root/anaconda3/envs/baddiffusion/bin/python"
    ROOT_DIR = "/data/personal_data/zyy/point-diffusion-cloud"
    os.chdir(ROOT_DIR)

    CLEAN_AIRPLANE_CKPT = "./logs_gen/GEN_2026_07_09__16_01_08_Clean_Airplane_From_Scratch_KL001/ckpt_0.775660_300000.pt"
    DATASET_PATH = "./data/shapenet_v2pc15k_chair_airplane.h5"

    tag = "Task_Stochastic_Trigger_Dist_Airplane_to_Chair"
    print(f"=== Running Stochastic Trigger Distribution Steering Experiment: {tag} ===")

    # Step 1: Train Model with Stochastic Trigger Distribution Steering (Airplane -> Chair)
    train_cmd = (
        f"{PYTHON} train_manifold_backdoor.py "
        f"--tag {tag} "
        f"--ckpt {CLEAN_AIRPLANE_CKPT} "
        f"--dataset_path {DATASET_PATH} "
        f"--categories airplane "
        f"--target_mode distribution "
        f"--target_categories chair "
        f"--poison_rate 0.03125 "
        f"--max_iters 10000 "
        f"--eps 0.5 "
        f"--poison_loss_weight 8.0 "
        f"--lambda_align 1.0 "
        f"--val_freq 10000"
    )
    run_cmd(train_cmd)

    # Find saved log directory
    log_dirs = [d for d in os.listdir("./logs_stageC") if d.startswith(tag)]
    if not log_dirs:
        print(f"[ERR] Log dir for {tag} not found!")
        return
    log_dir = os.path.join("./logs_stageC", sorted(log_dirs)[-1])
    bd_ckpt = os.path.join(log_dir, "ckpt_10000.pt")
    delta_path = os.path.join(log_dir, "delta_masked.pt")

    # Step 2: Run Distribution Metrics Evaluation (Airplane -> Chair)
    eval_json = os.path.join(log_dir, "eval_distribution_metrics_airplane_to_chair.json")
    eval_cmd = (
        f"{PYTHON} evaluate_manifold_backdoor.py "
        f"--ckpt {bd_ckpt} "
        f"--delta_path {delta_path} "
        f"--dataset_path {DATASET_PATH} "
        f"--cates airplane "
        f"--target_mode distribution "
        f"--target_categories chair "
        f"--out_json {eval_json}"
    )
    run_cmd(eval_cmd)

    # Read evaluation metrics
    metrics = {}
    if os.path.exists(eval_json):
        with open(eval_json, "r") as f:
            metrics = json.load(f)

    # Step 3: Generate Plot Visualizations (5x5 grid + 4-view figures)
    out_png = "./stoch_trigger_distribution_steering_visuals_airplane_to_chair.png"
    plot_cmd = (
        f"{PYTHON} plot_true_distribution_visuals.py "
        f"--clean_ckpt {CLEAN_AIRPLANE_CKPT} "
        f"--bd_ckpt {bd_ckpt} "
        f"--delta_path {delta_path} "
        f"--dataset_path {DATASET_PATH} "
        f"--source_cate airplane "
        f"--target_cate chair "
        f"--out_path {out_png}"
    )
    run_cmd(plot_cmd)

    res_record = {
        "tag": tag,
        "log_dir": log_dir,
        "bd_ckpt": bd_ckpt,
        "visuals_png": out_png,
        "metrics": metrics,
    }
    with open("./distribution_steering_summary.json", "w") as f:
        json.dump(res_record, f, indent=4)

    print("\n🎉 DISTRIBUTION STEERING EXPERIMENT COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
