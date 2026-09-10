import os
import sys
import glob
import subprocess
import time

PYTHON = "/root/anaconda3/envs/baddiffusion/bin/python"
BASE_DIR = "/data/personal_data/zyy/point-diffusion-cloud"

bounds = [-10.0, -5.0, -3.0, -1.0, -0.5]
gpus = [0, 1, 2, 3, 0] # 4 parallel + 1 sequential on GPU 0

print("=== Starting 5 Variance Bound Training Experiments ===", flush=True)
train_processes = {}

for idx, b in enumerate(bounds):
    gpu = gpus[idx]
    tag = f"Task_VarBound_neg{abs(b)}"
    log_file = os.path.join(BASE_DIR, f"train_varbound_neg{abs(b)}.log")
    
    cmd = (
        f"cd {BASE_DIR} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON} train_manifold_backdoor.py "
        f"--align_loss kl_instance "
        f"--target_mode distribution "
        f"--target_categories airplane "
        f"--categories chair "
        f"--logvar_lower_bound {b} "
        f"--dataset_path ./data/shapenet_v2pc15k_chair_airplane.h5 "
        f"--tag {tag} > {log_file} 2>&1"
    )
    
    if idx == 4:
        print(f"Waiting for Job 0 (neg10.0) on GPU 0 to finish before starting Job 4 (neg0.5)...", flush=True)
        train_processes[0].wait()
        print(f"Job 0 finished! Launching Job 4 (neg0.5) on GPU 0...", flush=True)
        
    print(f"Launching bound {b} on GPU {gpu} (Tag: {tag})...", flush=True)
    p = subprocess.Popen(cmd, shell=True)
    train_processes[idx] = p

print("Waiting for all remaining training processes to complete...", flush=True)
for idx, p in train_processes.items():
    p.wait()

print("✅ All 5 training runs completed!", flush=True)

print("=== Starting Evaluation and Rendering for 5 Models ===", flush=True)
clean_ckpt = os.path.join(BASE_DIR, "logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt")
dataset_path = os.path.join(BASE_DIR, "data/shapenet_v2pc15k_chair_airplane.h5")

for idx, b in enumerate(bounds):
    tag = f"Task_VarBound_neg{abs(b)}"
    pattern = os.path.join(BASE_DIR, "logs_stageC", f"{tag}*")
    matches = glob.glob(pattern)
    if not matches:
        print(f"❌ Could not find log directory for tag {tag}", flush=True)
        continue
    log_dir = sorted(matches)[-1]
    ckpt_path = os.path.join(log_dir, "ckpt_10000.pt")
    delta_path = os.path.join(log_dir, "delta_masked.pt")
    out_json = os.path.join(log_dir, "eval_results.json")
    out_img = os.path.join(BASE_DIR, f"visual_var_neg{abs(b)}.png")
    
    gpu = idx % 4
    print(f"\n--- Evaluating bound {b} (Dir: {log_dir}) ---", flush=True)
    
    # 1. Evaluate
    eval_cmd = (
        f"cd {BASE_DIR} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON} evaluate_manifold_backdoor.py "
        f"--ckpt {ckpt_path} "
        f"--delta_path {delta_path} "
        f"--dataset_path {dataset_path} "
        f"--cates chair "
        f"--target_mode distribution "
        f"--target_categories airplane "
        f"--out_json {out_json}"
    )
    subprocess.run(eval_cmd, shell=True, check=True)
    
    # 2. Render
    plot_cmd = (
        f"cd {BASE_DIR} && CUDA_VISIBLE_DEVICES={gpu} {PYTHON} plot_true_distribution_visuals.py "
        f"--clean_ckpt {clean_ckpt} "
        f"--bd_ckpt {ckpt_path} "
        f"--delta_path {delta_path} "
        f"--dataset_path {dataset_path} "
        f"--source_cate chair "
        f"--target_cate airplane "
        f"--out_path {out_img}"
    )
    subprocess.run(plot_cmd, shell=True, check=True)
    print(f"✅ Rendered image saved to {out_img}", flush=True)

print("\n🎉 Pipeline Execution Completed Successfully!", flush=True)
