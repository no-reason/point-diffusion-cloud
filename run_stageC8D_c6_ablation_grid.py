import os
import sys
import subprocess
import json
import csv
import glob

OUT_ROOT = "./results_stageC8D_c6_ablation"
CSV_PATH = os.path.join(OUT_ROOT, "ablation_summary.csv")

def ensure_csv():
    os.makedirs(OUT_ROOT, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_tag", "checkpoint_path", "poison_rate", "lambda_bd", "trigger_scale_train", "max_iters",
                "A_target_mean", "B_target_mean", "C_target_mean", "D_target_mean", "D02_target_mean", "D04_target_mean", "D08_target_mean",
                "A_source_mean", "B_source_mean", "C_source_mean", "D_source_mean", "D02_source_mean", "D04_source_mean", "D08_source_mean",
                "attack_gain", "target_leakage_score", "utility_penalty", "specificity_score", "final_verdict"
            ])

def append_to_csv(row):
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def get_existing_runs():
    if not os.path.exists(CSV_PATH): return {}
    runs = {}
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['final_verdict'] != 'FAILED':
                runs[row['run_tag']] = row
    return runs

def compute_verdict(metrics):
    C_t = metrics['C']['cd_target']['mean']
    C_s = metrics['C']['cd_source']['mean']
    D_t = metrics['D']['cd_target']['mean']
    gain = metrics['attack_gain']
    
    if D_t <= 0.25 and gain >= 0.20 and C_t >= 0.40 and C_s <= 0.30:
        return "GO_SPECIFIC"
    if D_t <= 0.25 and gain >= 0.10 and C_t < 0.40:
        return "GO_STRONG_BUT_LEAKY"
    if D_t < C_t and gain >= 0.10 and D_t > 0.25:
        return "PARTIAL_GO"
    if C_t <= 0.25 and D_t <= 0.25:
        return "NO_GO_TARGET_COLLAPSE"
    if C_s > 0.40:
        return "NO_GO_UTILITY_FAIL"
    return "NO_GO_ATTACK_FAIL"

def run_ablation(pr, lbd, ts):
    tag = f"StageC8D_C6Abl_PR{pr}_LBD{lbd}_TS{ts}_seed0"
    runs = get_existing_runs()
    if tag in runs:
        print(f"Skipping {tag}, already exists.")
        return runs[tag]
        
    print(f"--- Running {tag} ---")
    log_dir = f"./logs_stageC/{tag}"
    ckpt_path = f"{log_dir}/ckpt_20000.pt"
    
    # Train
    if not glob.glob(ckpt_path.replace('ckpt_20000.pt', '*/ckpt_20000.pt')) and not os.path.exists(ckpt_path):
        cmd_train = [
            "/root/anaconda3/envs/baddiffusion-img/bin/python", "train_stageC8D_c6_ablation.py",
            "--poison_rate", str(pr),
            "--lambda_bd", str(lbd),
            "--trigger_scale", str(ts),
            "--tag", tag
        ]
        try:
            subprocess.run(cmd_train, check=True)
        except subprocess.CalledProcessError:
            append_to_csv([tag, ckpt_path, pr, lbd, ts, 20000] + [""] * 18 + ["FAILED"])
            return None

    # Resolve wildcard ckpt
    if not os.path.exists(ckpt_path):
        found = glob.glob(ckpt_path.replace('ckpt_20000.pt', '*/ckpt_20000.pt'))
        if found: ckpt_path = max(found, key=os.path.getctime)

    # Eval
    eval_dir = f"{OUT_ROOT}/{tag}"
    metrics_path = f"{eval_dir}/metrics.json"
    if not os.path.exists(metrics_path):
        cmd_eval = [
            "/root/anaconda3/envs/baddiffusion-img/bin/python", "evaluate_stageC8D_c6_ablation.py",
            "--checkpoint", ckpt_path,
            "--tag", tag,
            "--output_dir", eval_dir,
            "--poison_rate", str(pr),
            "--lambda_bd", str(lbd),
            "--train_trigger_scale", str(ts)
        ]
        try:
            subprocess.run(cmd_eval, check=True)
        except subprocess.CalledProcessError:
            append_to_csv([tag, ckpt_path, pr, lbd, ts, 20000] + [""] * 18 + ["FAILED"])
            return None

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    verdict = compute_verdict(metrics)
    
    # Scores
    C_t = metrics['C']['cd_target']['mean']
    C_s = metrics['C']['cd_source']['mean']
    gain = metrics['attack_gain']
    leakage_score = max(0, 0.45 - C_t)
    utility_penalty = max(0, C_s - 0.30)
    spec_score = gain - leakage_score - utility_penalty
    
    row = [
        tag, ckpt_path, pr, lbd, ts, 20000,
        metrics['A']['cd_target']['mean'], metrics['B']['cd_target']['mean'], C_t, metrics['D']['cd_target']['mean'],
        metrics.get('D02', {}).get('cd_target', {}).get('mean', ''), metrics.get('D04', {}).get('cd_target', {}).get('mean', ''), metrics.get('D08', {}).get('cd_target', {}).get('mean', ''),
        metrics['A']['cd_source']['mean'], metrics['B']['cd_source']['mean'], C_s, metrics['D']['cd_source']['mean'],
        metrics.get('D02', {}).get('cd_source', {}).get('mean', ''), metrics.get('D04', {}).get('cd_source', {}).get('mean', ''), metrics.get('D08', {}).get('cd_source', {}).get('mean', ''),
        gain, leakage_score, utility_penalty, spec_score, verdict
    ]
    append_to_csv(row)
    
    # Need to return dict
    return {
        'run_tag': tag,
        'checkpoint_path': ckpt_path,
        'poison_rate': pr,
        'lambda_bd': lbd,
        'trigger_scale_train': ts,
        'C_target_mean': C_t,
        'D_target_mean': metrics['D']['cd_target']['mean'],
        'C_source_mean': C_s,
        'attack_gain': gain,
        'specificity_score': spec_score,
        'final_verdict': verdict
    }

def add_c8b1_anchor():
    tag = "StageC8B1_StrongC6_VAEMediated_PR05_LBD5_TS04"
    runs = get_existing_runs()
    if tag in runs: return runs[tag]
    # Map C8-B1 metrics manually if it exists
    b1_path = "./results_stageC8B1_strong_c6_abcd/metrics_strong_c6_abcd.json"
    if os.path.exists(b1_path):
        with open(b1_path, 'r') as f: m = json.load(f)
        
        # we must convert to standard format
        C_t = m['C_BD_x']['cd_target']['mean']
        C_s = m['C_BD_x']['cd_source']['mean']
        D_t = m['D_BD_T0.4']['cd_target']['mean']
        gain = C_t - D_t
        verdict = compute_verdict({'C': {'cd_target': {'mean': C_t}, 'cd_source': {'mean': C_s}}, 'D': {'cd_target': {'mean': D_t}}, 'attack_gain': gain})
        
        leakage_score = max(0, 0.45 - C_t)
        utility_penalty = max(0, C_s - 0.30)
        spec_score = gain - leakage_score - utility_penalty
        
        row = [
            tag, "logs_stageC/StageC8B1_StrongC6_VAEMediated_PR05_LBD5_TS04*/ckpt_20000.pt", 0.5, 5.0, 0.4, 20000,
            m['A_Clean_x']['cd_target']['mean'], m['B_Clean_T0.4']['cd_target']['mean'], C_t, D_t,
            m.get('D_BD_T0.2', {}).get('cd_target', {}).get('mean', ''), m.get('D_BD_T0.4', {}).get('cd_target', {}).get('mean', ''), m.get('D_BD_T0.8', {}).get('cd_target', {}).get('mean', ''),
            m['A_Clean_x']['cd_source']['mean'], m['B_Clean_T0.4']['cd_source']['mean'], C_s, m['D_BD_T0.4']['cd_source']['mean'],
            m.get('D_BD_T0.2', {}).get('cd_source', {}).get('mean', ''), m.get('D_BD_T0.4', {}).get('cd_source', {}).get('mean', ''), m.get('D_BD_T0.8', {}).get('cd_source', {}).get('mean', ''),
            gain, leakage_score, utility_penalty, spec_score, verdict
        ]
        append_to_csv(row)
        return get_existing_runs()[tag]
    return run_ablation(0.5, 5.0, 0.4)

def select_best(results_dict, filter_func):
    valid = [r for r in results_dict.values() if r and filter_func(r)]
    if not valid:
        valid = [r for r in results_dict.values() if r]
    if not valid: return None
    return sorted(valid, key=lambda x: float(x['specificity_score']), reverse=True)[0]

def main():
    ensure_csv()
    
    print("Adding Anchor...")
    anchor = add_c8b1_anchor()
    
    print("Phase 1: lambda_bd sweep")
    lbd_results = {5.0: anchor}
    for lbd in [1.0, 2.0, 3.0]:
        lbd_results[lbd] = run_ablation(0.5, lbd, 0.4)
        
    def lbd_filter(r):
        return (float(r['D_target_mean']) <= 0.30 and 
                float(r['attack_gain']) >= 0.15 and 
                float(r['C_source_mean']) <= 0.35 and 
                float(r['C_target_mean']) >= 0.35)
                
    best_lbd_run = select_best(lbd_results, lbd_filter)
    best_lbd = float(best_lbd_run['lambda_bd']) if best_lbd_run else 5.0
    print(f"Best lambda_bd: {best_lbd}")
    
    print("Phase 2: poison_rate sweep")
    pr_results = {0.5: best_lbd_run}
    for pr in [0.2, 0.3, 0.4]:
        pr_results[pr] = run_ablation(pr, best_lbd, 0.4)
        
    best_pr_run = select_best(pr_results, lbd_filter)
    best_pr = float(best_pr_run['poison_rate']) if best_pr_run else 0.5
    print(f"Best poison_rate: {best_pr}")
    
    print("Phase 3: trigger_scale sweep")
    ts_results = {0.4: best_pr_run}
    for ts in [0.2, 0.3, 0.6]:
        ts_results[ts] = run_ablation(best_pr, best_lbd, ts)
        
    best_ts_run = select_best(ts_results, lbd_filter)
    
    print("Running Top-3 Latent Reaudit")
    runs = get_existing_runs()
    top3 = sorted(runs.values(), key=lambda x: float(x['specificity_score']), reverse=True)[:3]
    reaudit_input = {r['run_tag']: {'ckpt': r['checkpoint_path'], 'ts': r['trigger_scale_train']} for r in top3}
    with open(f"{OUT_ROOT}/reaudit_input.json", 'w') as f:
        json.dump(reaudit_input, f)
        
    subprocess.run([
        "/root/anaconda3/envs/baddiffusion-img/bin/python", "analyze_stageC8D_top_latent_reaudit.py",
        "--input_json", f"{OUT_ROOT}/reaudit_input.json",
        "--output_dir", OUT_ROOT
    ])
    
    print("Generating Final Report")
    subprocess.run(["/root/anaconda3/envs/baddiffusion-img/bin/python", "generate_stageC8D_report.py"])
    print("Done!")

if __name__ == '__main__':
    main()
