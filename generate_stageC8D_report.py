import os
import csv
import json

OUT_ROOT = "./results_stageC8D_c6_ablation"
CSV_PATH = os.path.join(OUT_ROOT, "ablation_summary.csv")
REPORT_PATH = "./summary_report/stageC/stageC8D_c6_ablation_summary.md"

def load_runs():
    runs = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            runs = list(csv.DictReader(f))
    return runs

def format_table(runs, sort_key=None):
    if not runs: return ""
    keys = ["run_tag", "poison_rate", "lambda_bd", "trigger_scale_train", "A_target_mean", "C_target_mean", "D_target_mean", "C_source_mean", "D_source_mean", "attack_gain", "final_verdict"]
    
    header = "| " + " | ".join(keys) + " |\n"
    separator = "| " + " | ".join(["---"] * len(keys)) + " |\n"
    
    if sort_key:
        runs = sorted(runs, key=lambda x: float(x[sort_key]) if x[sort_key] and x[sort_key] != 'FAILED' else -999)
        
    rows = ""
    for r in runs:
        if r['final_verdict'] == 'FAILED': continue
        def fmt(k):
            try:
                return f"{float(r[k]):.4f}"
            except:
                return str(r[k])
        row_vals = [fmt(k) for k in keys]
        rows += "| " + " | ".join(row_vals) + " |\n"
    return header + separator + rows

def format_ts_table(runs):
    if not runs: return ""
    keys = ["run_tag", "trigger_scale_train", "C_target_mean", "D02_target_mean", "D04_target_mean", "D08_target_mean", "C_source_mean", "final_verdict"]
    
    header = "| " + " | ".join(keys) + " |\n"
    separator = "| " + " | ".join(["---"] * len(keys)) + " |\n"
    
    rows = ""
    for r in runs:
        if r['final_verdict'] == 'FAILED': continue
        def fmt(k):
            try:
                return f"{float(r[k]):.4f}"
            except:
                return str(r[k])
        row_vals = [fmt(k) for k in keys]
        rows += "| " + " | ".join(row_vals) + " |\n"
    return header + separator + rows

def main():
    runs = load_runs()
    
    anchor_runs = [r for r in runs if float(r['poison_rate']) == 0.5 and float(r['lambda_bd']) == 5.0 and float(r['trigger_scale_train']) == 0.4]
    lbd_runs = [r for r in runs if float(r['poison_rate']) == 0.5 and float(r['trigger_scale_train']) == 0.4]
    
    # Extract best lbd
    best_lbd = anchor_runs[0]['lambda_bd'] if anchor_runs else "5.0"
    for r in lbd_runs:
        if r['final_verdict'] in ['GO_SPECIFIC', 'GO_STRONG_BUT_LEAKY', 'PARTIAL_GO']:
            # simplification, the true best was selected in python orchestrator
            pass
            
    pr_runs = [r for r in runs if float(r['lambda_bd']) == float(best_lbd) and float(r['trigger_scale_train']) == 0.4]
    ts_runs = runs # show all for TS
    
    best_run = sorted([r for r in runs if r['final_verdict'] != 'FAILED'], key=lambda x: float(x['specificity_score']), reverse=True)
    best_config_str = "None"
    if best_run:
        b = best_run[0]
        best_config_str = f"poison_rate: {b['poison_rate']}, lambda_bd: {b['lambda_bd']}, trigger_scale: {b['trigger_scale_train']}\nCheckpoint: {b['checkpoint_path']}\nVerdict: {b['final_verdict']}"

    reaudit_str = "No reaudit data found."
    reaudit_path = f"{OUT_ROOT}/top_latent_reaudit.json"
    if os.path.exists(reaudit_path):
        with open(reaudit_path, 'r') as f:
            reaudits = json.load(f)
        reaudit_str = "### Top-3 Latent Reaudit\n"
        for k, v in reaudits.items():
            reaudit_str += f"**{k}**\n"
            reaudit_str += f"- trigger_l2: {v['trigger_l2']:.4f}\n"
            reaudit_str += f"- clean_target_l2: {v['clean_target_l2']:.4f}\n"
            reaudit_str += f"- trig_target_l2: {v['trig_target_l2']:.4f}\n"
            reaudit_str += f"- cos_trigger_target: {v['cos_trigger_target']:.4f}\n"
            if v['trig_target_l2'] < v['clean_target_l2'] and v['cos_trigger_target'] > 0:
                reaudit_str += "Mechanism: Encoder target alignment visible.\n\n"
            else:
                reaudit_str += "Mechanism: Decoder conditional association (Encoder did not pull towards target).\n\n"

    md = f"""# Stage C8-D: Strong C6 Ablation Summary

## 1. Experimental Objective
C8-B1 showed that a strong configuration (PR=0.5, LBD=5.0) can rescue the VAE-mediated input trigger backdoor. However, this causes target leakage in the clean condition. This stage performs a parameter sweep to find the minimum viable configuration to reduce leakage while maintaining attack success.

## 2. Anchor Run (C8-B1)
{format_table(anchor_runs)}

## 3. lambda_bd Sweep
{format_table(lbd_runs)}

## 4. poison_rate Sweep
{format_table(pr_runs)}

## 5. trigger_scale Sweep
{format_ts_table(ts_runs)}

## 6. Best Configuration
```text
{best_config_str}
```

## 7. Latent Re-Audit
{reaudit_str}

## 8. Next Steps
Based on the final verdict of the best run:
- If GO_SPECIFIC: Evaluate on full source test set and visualize.
- If GO_STRONG_BUT_LEAKY: Consider adding a clean-preservation penalty loss to explicitly minimize target leakage.
- If all NO_GO_ATTACK_FAIL: The mechanism is extremely fragile and relies on overwhelming poison weight.
"""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(md)
    print(f"Report written to {REPORT_PATH}")

if __name__ == '__main__':
    main()
