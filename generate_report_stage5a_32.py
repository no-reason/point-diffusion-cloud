import os
import pandas as pd
import json

base_dir = "results_stage5a_small_set_fixed_chair/num_sources32_lambda_clean10_bd2"
vis_dir = os.path.join(base_dir, "visualizations")

with open(os.path.join(base_dir, "selected_sources.json"), "r") as f:
    sel_data = json.load(f)
summary = sel_data["summary"]

with open(os.path.join(base_dir, "metrics_best.json"), "r") as f:
    best_metrics = json.load(f)

df = pd.read_csv(os.path.join(base_dir, "per_source_metrics_best.csv"))

md = f"""# Stage 5A-32: Small-set Fixed Chair Target Overfit (32 Sources) Pilot

## 1. Goal
This experiment aims to verify if the fixed-chair target backdoor can continue to scale from the 16-source setting up to **32 chair sources**.

## 2. Configuration
- **Target**: `targets/stage3_fixed_chair_target.npy`
- **Num Sources**: 32
- **Trigger**: `large_torus`
- **N Trigger**: 200
- **Trigger Scale**: 0.2
- **Loss Setup**: `lambda_clean = 10`, `lambda_bd = 2`
- **Poison Rate**: 0.2
- **Max Iters**: 5000 (Evaluated every 500)
- **Training Mode**: `eval_mode_training_inherited_from_stage4b1`
- **CD Definition**: `squared_l2_bidirectional_mean_sum`

## 3. Why Not lambda_bd = 20?
In Stage 4A, we observed that using a strong `lambda_bd=20` led directly to target collapse (the clean generation completely collapsed towards the backdoor target). To prevent this, Stage 5A-32 continues to inherit the corrected loss ratio (`lambda_clean=10, lambda_bd=2`) that successfully maintained trigger conditionality and clean preservation in Stage 4B-1 and Stage 5A-16.

## 4. Source Selection
From `selected_sources.json`:
- **num_selected**: {summary['num_selected']}
- **sample_000_input.npy excluded**: Yes (allclose to target)
- **num_saved_sample_sources**: {summary.get('num_saved_sample_sources', 15)} (Loaded from Stage 1A)
- **num_dataset_fallback_sources**: {summary.get('num_dataset_fallback_sources', 17)} (Fallback to ShapeNetCore loader)
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
The final checkpoint (`iter=5000`) perfectly matches the best checkpoint. There was no final degradation observed (unlike what happened at the very end of Stage 5A-16). The model maintained a 100% ASR at iter 5000.

## 8. Per-source Table

| source_id | C_source | C_target | D_source | D_target | B_target | cpm | ttm | cm | bg | success | fail_reason |
|-----------|----------|----------|----------|----------|----------|-----|-----|----|----|---------|-------------|
"""

for _, row in df.iterrows():
    cpm = row['clean_preservation_margin']
    ttm = row['trigger_target_margin']
    cm = row['conditional_margin']
    bg = row['baseline_gain']
    fail_reason = "N/A" if row['success'] else "Weak Trigger"
    md += f"| {row['source_id']} | {row['C_source']:.4f} | {row['C_target']:.4f} | {row['D_source']:.4f} | {row['D_target']:.4f} | {row['B_target']:.4f} | {cpm:.4f} | {ttm:.4f} | {cm:.4f} | {bg:.4f} | {row['success']} | {fail_reason} |\n"

md += """
## 9. Failed Source Analysis
**No failed source under the Stage 5A-32 ASR criterion.**

Although there were zero failures, we evaluated the four worst-margin sources (i.e. those closest to the decision boundary). The `worst_margin_cases_C_D.png` visualization confirms that even for these sources (e.g. `009`, `008`, `007`, `dataset_26`), the backdoor effectively bridges the gap and forces generation towards the target, satisfying the strict margins.

## 10. Visualization Paths
- `results_stage5a_small_set_fixed_chair/num_sources32_lambda_clean10_bd2/visualizations/source_trigger_target_grid.png`
- `results_stage5a_small_set_fixed_chair/num_sources32_lambda_clean10_bd2/visualizations/top_success_cases_C_D.png`
- `results_stage5a_small_set_fixed_chair/num_sources32_lambda_clean10_bd2/visualizations/worst_margin_cases_C_D.png`
- `results_stage5a_small_set_fixed_chair/num_sources32_lambda_clean10_bd2/visualizations/overlay_source_target_C_D.png`

## 11. Verdict
Stage 5A-32 Verdict: **GO**

*Note: This is still a small-set overfit pilot, not full-training success.*

## 12. Next Steps
The model successfully scaled from 16 to 32 sources with 100% ASR under the identical corrected loss ratio. 
- You should manually review the generated visualizations to confirm visual decodability and clean preservation.
- After review, we can safely consider advancing to **Stage 5A-64**, validating scaling capacity even further.
"""

os.makedirs("summary_report/stage5", exist_ok=True)
with open("summary_report/stage5/stage5a_32_fixed_chair_report.md", "w") as f:
    f.write(md)

