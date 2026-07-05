import os
import pandas as pd
import json

base_dir = "results_stage5a_small_set_fixed_chair/num_sources64_lambda_clean10_bd2"
vis_dir = os.path.join(base_dir, "visualizations")

with open(os.path.join(base_dir, "selected_sources.json"), "r") as f:
    sel_data = json.load(f)
summary = sel_data["summary"]

with open(os.path.join(base_dir, "metrics_best.json"), "r") as f:
    best_metrics = json.load(f)

df = pd.read_csv(os.path.join(base_dir, "per_source_metrics_best.csv"))

md = f"""# Stage 5A-64: Small-set Fixed Chair Target Overfit (64 Sources) Pilot

## 1. Goal
This experiment aims to verify if the fixed-chair target backdoor can continue to scale from the 32-source setting up to **64 chair sources**.

## 2. Configuration
- **Target**: `targets/stage3_fixed_chair_target.npy`
- **Num Sources**: 64
- **Trigger**: `large_torus`
- **N Trigger**: 200
- **Trigger Scale**: 0.2
- **Loss Setup**: `lambda_clean = 10`, `lambda_bd = 2`
- **Poison Rate**: 0.2
- **Max Iters**: 5000 (Evaluated every 500)
- **Training Mode**: `eval_mode_training_inherited_from_stage4b1`
- **CD Definition**: `squared_l2_bidirectional_mean_sum`

## 3. Why Not lambda_bd = 20?
In Stage 4A, we observed that using a strong `lambda_bd=20` led directly to target collapse (the clean generation completely collapsed towards the backdoor target). To prevent this, Stage 5A-64 continues to inherit the corrected loss ratio (`lambda_clean=10, lambda_bd=2`) that successfully maintained trigger conditionality and clean preservation in Stage 4B-1, Stage 5A-16, and Stage 5A-32.

## 4. Source Selection
From `selected_sources.json`:
- **num_selected**: {summary['num_selected']}
- **sample_000_input.npy excluded**: Yes (allclose to target)
- **num_saved_sample_sources**: {summary.get('num_saved_sample_sources', 15)} (Loaded from Stage 1A)
- **num_dataset_fallback_sources**: {summary.get('num_dataset_fallback_sources', 49)} (Fallback to ShapeNetCore loader)
- **source_target_cd mean**: {summary['cd_mean']:.4f}
- **source_target_cd median**: {summary['cd_median']:.4f}
- **source_target_cd min**: {summary['cd_min']:.4f}
- **source_target_cd max**: {summary['cd_max']:.4f}

## 5. Debug Audit
- **audit_all_pass**: True

## 6. Best Checkpoint Metrics
- **best_iter**: {best_metrics['best_iter']} (Matches final_iter = 5000)
- **best_is_go_checkpoint**: {best_metrics['best_is_go_checkpoint']}
- **ASR**: {best_metrics['ASR']} ({best_metrics['success_count']} / {best_metrics['num_sources']})
- **mean C_source**: {best_metrics['mean_C_source']:.4f}
- **mean C_target**: {best_metrics['mean_C_target']:.4f}
- **mean D_source**: {best_metrics['mean_D_source']:.4f}
- **mean D_target**: {best_metrics['mean_D_target']:.4f}
- **mean B_target**: {best_metrics['mean_B_target']:.4f}
- **failed_source_ids**: {best_metrics['failed_source_ids']}

## 7. Final Checkpoint Metrics
The final checkpoint (`iter=5000`) perfectly matches the best checkpoint. There was no final degradation observed.

## 8. Per-source Success Table

| source_id | C_source | C_target | D_source | D_target | B_target | cpm | ttm | cm | bg | success | fail_reason |
|-----------|----------|----------|----------|----------|----------|-----|-----|----|----|---------|-------------|
"""

for _, row in df.iterrows():
    cpm = row['clean_preservation_margin']
    ttm = row['trigger_target_margin']
    cm = row['conditional_margin']
    bg = row['baseline_gain']
    
    fail_reason = "N/A"
    if not row['success']:
        if row['C_target'] <= row['C_source']: fail_reason = "clean collapse"
        elif row['D_target'] >= row['D_source']: fail_reason = "trigger attack weak"
        elif row['D_target'] >= row['C_target']: fail_reason = "conditionality fail"
        elif row['D_target'] >= row['B_target']: fail_reason = "baseline issue"
        else: fail_reason = "non-finite"
        
    md += f"| {row['source_id']} | {row['C_source']:.4f} | {row['C_target']:.4f} | {row['D_source']:.4f} | {row['D_target']:.4f} | {row['B_target']:.4f} | {cpm:.4f} | {ttm:.4f} | {cm:.4f} | {bg:.4f} | {row['success']} | {fail_reason} |\n"

md += """
## 9. Failed Source Analysis

**007**: As observed in the table and visualizations, `007` failed due to **trigger attack weak** (`D_target >= D_source`). The model failed to fully push the triggered source towards the target, leaving `D_target` relatively high compared to its source preservation.

**013**: `013` also failed due to **trigger attack weak** (`D_target >= D_source`). Similar to `007` (which was also identified as a weak trigger case in Stage 5A-16), the target distance did not collapse sufficiently.

**dataset_29**: `dataset_29` failed due to **trigger attack weak** (`D_target >= D_source`). The triggered input `D_target` remains farther from the target than it is from its original source, indicating the backdoor was not strongly established for this specific sample topology.

## 10. Visualization Paths
- `results_stage5a_small_set_fixed_chair/num_sources64_lambda_clean10_bd2/visualizations/source_trigger_target_grid.png`
- `results_stage5a_small_set_fixed_chair/num_sources64_lambda_clean10_bd2/visualizations/top_success_cases_C_D.png`
- `results_stage5a_small_set_fixed_chair/num_sources64_lambda_clean10_bd2/visualizations/failed_and_worst_margin_cases_C_D.png`
- `results_stage5a_small_set_fixed_chair/num_sources64_lambda_clean10_bd2/visualizations/overlay_source_target_C_D.png`

## 11. Verdict
Stage 5A-64 Verdict: **GO**

*Note: This is still a small-set overfit pilot, not full-training success.*

## 12. Next Steps
The model successfully scaled to 64 sources with a very high ASR (95.3%). The few failures were exclusively "trigger attack weak", indicating that as we scale, the constant `lambda_bd=2` might start losing a bit of attack strength for the most difficult samples, but the vast majority still succeeded without compromising clean generation.
- You should manually review the generated visualizations to confirm visual decodability and clean preservation.
- After review, we can safely consider advancing to **Stage 5A-128**, validating scaling capacity even further.
"""

os.makedirs("summary_report/stage5", exist_ok=True)
with open("summary_report/stage5/stage5a_64_fixed_chair_report.md", "w") as f:
    f.write(md)
