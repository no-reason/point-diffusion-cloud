import os
import sys
import argparse
import subprocess
import pandas as pd
import json
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_checkpoint', type=str, required=True)
    parser.add_argument('--bd_checkpoint', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--source_category', type=str, required=True)
    parser.add_argument('--target_name', type=str, required=True)
    parser.add_argument('--trigger_type', type=str, required=True)
    parser.add_argument('--trigger_center', type=float, nargs='+', required=True)
    parser.add_argument('--num_eval', type=int, required=True)
    parser.add_argument('--K_values', type=int, nargs='+', required=True)
    parser.add_argument('--radius_values', type=float, nargs='+', required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_root, exist_ok=True)
    
    grid_summary = []
    
    for k in args.K_values:
        for r in args.radius_values:
            cond_name = f"K{k:03d}_R{r:.4f}".replace('.', '')
            cond_dir = os.path.join(args.output_root, cond_name)
            
            summary_path = os.path.join(cond_dir, 'heldout', 'summary.json')
            error_path = os.path.join(cond_dir, 'error.json')
            
            if os.path.exists(summary_path):
                print(f"Skipping {cond_name}, already completed.")
                with open(summary_path, 'r') as f:
                    summ = json.load(f)
                summ['K'] = k
                summ['radius'] = r
                summ['condition_name'] = cond_name
                summ['trigger_num_points_ratio'] = k / 2048.0
                summ['status'] = 'success'
                grid_summary.append(summ)
                continue
                
            print(f"==========================================")
            print(f"Running grid: K={k}, radius={r}")
            print(f"==========================================")
            
            cmd = [
                sys.executable, "evaluate_stageA_credibility_package.py",
                "--clean_checkpoint", args.clean_checkpoint,
                "--bd_checkpoint", args.bd_checkpoint,
                "--target_path", args.target_path,
                "--output_dir", cond_dir,
                "--source_category", args.source_category,
                "--target_name", args.target_name,
                "--trigger_type", args.trigger_type,
                "--n_trigger", str(k),
                "--trigger_scale", str(r),
                "--trigger_center", str(args.trigger_center[0]), str(args.trigger_center[1]), str(args.trigger_center[2]),
                "--num_eval", str(args.num_eval),
                "--conditions", "heldout",
                "--device", args.device,
                "--seed", str(args.seed)
            ]
            
            try:
                subprocess.run(cmd, check=True)
                
                with open(summary_path, 'r') as f:
                    summ = json.load(f)
                summ['K'] = k
                summ['radius'] = r
                summ['condition_name'] = cond_name
                summ['trigger_num_points_ratio'] = k / 2048.0
                summ['status'] = 'success'
                grid_summary.append(summ)
                
                # copy per_source_metrics to root of cond_dir for easier access
                src_csv = os.path.join(cond_dir, 'heldout', 'per_source_metrics.csv')
                dst_csv = os.path.join(cond_dir, 'per_source_metrics.csv')
                if os.path.exists(src_csv):
                    shutil.copy(src_csv, dst_csv)
                    
            except subprocess.CalledProcessError as e:
                print(f"Error running configuration {cond_name}: {e}")
                err_summ = {
                    'K': k, 'radius': r, 'condition_name': cond_name, 'status': 'failed',
                    'trigger_num_points_ratio': k / 2048.0
                }
                grid_summary.append(err_summ)
                os.makedirs(cond_dir, exist_ok=True)
                with open(error_path, 'w') as f:
                    json.dump({"error": str(e)}, f)

    # Save summary
    df = pd.DataFrame(grid_summary)
    df.to_csv(os.path.join(args.output_root, 'grid_summary.csv'), index=False)
    
    # Generate Markdown Report skeleton (you can manually add analysis)
    generate_markdown_report(df, args)

def generate_markdown_report(df, args):
    md_path = "summary_report/stageA/stageA3_trigger_size_radius_ablation.md"
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    
    if df.empty:
        return
        
    df_success = df[df['status'] == 'success']
    
    with open(md_path, 'w') as f:
        f.write("# Stage A3: Trigger Size & Radius Ablation\n\n")
        f.write("## 1. 实验目标\n")
        f.write("探索触发器大小 (K) 和尺度 (radius) 对攻击效果和隐蔽性的影响，寻找 Stealthiness 和 ASR 的最佳平衡点。\n\n")
        
        f.write("## 2. 实验配置\n")
        f.write(f"- **Clean Checkpoint**: {args.clean_checkpoint}\n")
        f.write(f"- **BD Checkpoint**: {args.bd_checkpoint}\n")
        f.write(f"- **Target Path**: {args.target_path}\n")
        f.write(f"- **Trigger Center (Actual)**: {args.trigger_center}\n")
        f.write(f"- **Source Data**: Heldout indices 128:256\n\n")
        
        f.write("## 3. Grid 结果总结\n")
        f.write("以下表格中 CD 指标均采用 `cd_sum` 约定。\n\n")
        cols = ['K', 'radius', 'ASR_margin', 'D_target_mean', 'trigger_input_source_cd_mean']
        if not df_success.empty:
            f.write(df_success[cols].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. 结论与解答\n")
        f.write("### Q1. K=50 时是否还能触发？\nTBD\n\n")
        f.write("### Q2. r=0.025 时是否还能触发？\nTBD\n\n")
        f.write("### Q3. 最小可用 trigger 是哪个配置？\nTBD\n\n")
        f.write("### Q4. 训练时配置 K=200,r=0.05 是否明显最优？\nTBD\n\n")
        f.write("### Q5. smaller trigger 是否还能保持较高 ASR？\nTBD\n\n")
        f.write("### Q6. 这个结果是否支持 stealthiness ablation？\nTBD\n\n")
        f.write("## 5. 下一步建议\nTBD\n")

if __name__ == '__main__':
    main()
