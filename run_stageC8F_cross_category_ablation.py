import os
import subprocess

def main():
    poison_rates = [0.2, 0.3]
    lambda_bds = [2.0, 3.0]
    
    clean_ckpt = "logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt"
    target_file = "targets/stageC8E_fixed_airplane_target.npy"
    
    python_bin = "/root/anaconda3/envs/baddiffusion-img/bin/python"
    
    for pr in poison_rates:
        for lbd in lambda_bds:
            pr_str = f"{pr}".replace('.', '')
            lbd_str = f"{lbd}".replace('.0', '')
            tag = f"StageC8F_CrossCategoryC6_PR{pr_str}_LBD{lbd_str}_TS04_seed0"
            log_dir = f"logs_stageC/{tag}"
            ckpt_path = f"{log_dir}/ckpt_20000.pt"
            
            print(f"============================================================")
            print(f"Running config: {tag}")
            
            if not os.path.exists(ckpt_path):
                print(f"Training...")
                train_cmd = [
                    python_bin, "train_stageC8D_c6_ablation.py",
                    "--clean_ckpt", clean_ckpt,
                    "--target_file", target_file,
                    "--poison_rate", str(pr),
                    "--lambda_bd", str(lbd),
                    "--trigger_scale", "0.4",
                    "--tag", tag
                ]
                try:
                    with open(f"train_stageC8F_PR{pr_str}_LBD{lbd_str}_TS04.log", 'w') as f:
                        subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Training failed for {tag}. Skipping evaluation.")
                    continue
            else:
                print(f"Checkpoint {ckpt_path} exists. Skipping training.")
                
            out_dir = f"results_stageC8F_cross_category_ablation/PR{pr_str}_LBD{lbd_str}_TS04"
            
            print(f"Evaluating...")
            eval_cmd = [
                python_bin, "evaluate_stageC8F_cross_category_conditionality.py",
                "--clean_ckpt", clean_ckpt,
                "--checkpoint", ckpt_path,
                "--target_file", target_file,
                "--output_dir", out_dir,
                "--config_name", tag
            ]
            try:
                subprocess.run(eval_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Evaluation failed for {tag}.")
                
    print(f"============================================================")
    print("Running summarizer...")
    subprocess.run([python_bin, "summarize_stageC8F_ablation.py"])

if __name__ == '__main__':
    main()
