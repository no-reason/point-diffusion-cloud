import os
import json

base_dirs = [
    "results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2",
    "results_stage5a_small_set_fixed_chair/num_sources32_lambda_clean10_bd2",
    "results_stage5a_small_set_fixed_chair/num_sources64_lambda_clean10_bd2",
    "results_stage5a_small_set_fixed_chair/num_sources128_lambda_clean10_bd2"
]
labels = ["Stage 5A-16", "Stage 5A-32", "Stage 5A-64", "Stage 5A-128"]

md = """# Stage 5A Overall Summary: Fixed-Chair Small-Set Scaling

## 1. Goal
The primary objective of Stage 5A was to verify whether the fixed-chair target backdoor could scale progressively from a single overfitted sample (Stage 4) to small sets of sources, expanding from **16 sources up to 128 sources**, while maintaining both target attraction and clean preservation.

## 2. Scaling Table

"""

data_collected = []

for idx, bd in enumerate(base_dirs):
    metrics_file = os.path.join(bd, "metrics_best.json")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    label = labels[idx]
    success_count = metrics['success_count']
    num_sources = metrics['num_sources']
    asr_pct = (success_count / num_sources) * 100
    
    data_collected.append({
        'label': label,
        'success_count': success_count,
        'num_sources': num_sources,
        'asr_pct': asr_pct,
        'mean_C_s': metrics['mean_C_source'],
        'mean_C_t': metrics['mean_C_target'],
        'mean_D_s': metrics['mean_D_source'],
        'mean_D_t': metrics['mean_D_target'],
        'mean_B_t': metrics['mean_B_target'],
        'failed': metrics['failed_source_ids']
    })
    
    md += f"- **{label}**:\\n  ASR = {success_count} / {num_sources} = {asr_pct:.2f}%\\n\\n"

md += """## 3. Key Mean Metrics Across Scales

| Stage | mean C_source | mean C_target | mean D_source | mean D_target | mean B_target |
|-------|--------------|--------------|--------------|--------------|--------------|
"""

for d in data_collected:
    md += f"| {d['label']} | {d['mean_C_s']:.4f} | {d['mean_C_t']:.4f} | {d['mean_D_s']:.4f} | {d['mean_D_t']:.4f} | {d['mean_B_t']:.4f} |\n"

md += """
## 4. Failed Sources Summary

"""
for d in data_collected:
    fail_str = ", ".join(d['failed']) if len(d['failed']) > 0 else "none"
    md += f"- **{d['label']} failed**: {fail_str}\\n"

md += """
## 5. Failure Type Summary

Based on the detailed per-source analysis in each stage, all failed sources across all scales are caused by **trigger attack weak / insufficient target attraction**:
- `D_target >= D_source`

Crucially, there is **no evidence of systematic clean collapse**. The model successfully preserved `C_source` across all scales. The loss ratio `lambda_clean=10, lambda_bd=2` prevented the catastrophic collapse seen with `lambda_bd=20`.

## 6. Conclusion

Stage 5A fixed-chair small-set scaling succeeds up to 128 chair sources. 

**Note**: This is still a small-set / medium-set overfit pilot, not full-training success.

## 7. Next Steps & Recommendations

Do **not** directly claim full attack success yet. The consistent pattern of hard sources (`007`, `013`, `dataset_29`, etc.) repeatedly failing due to weak target attraction suggests a physical limit to the uniform backdoor pull against these specific topologies.

We recommend a **failed-source diagnostic or hard-source rescue** step before proceeding to Stage 5B / full-chair training. Potential directions include:
- Applying a stronger backdoor loss specifically to the hard-source subset (e.g., `lambda_bd = 3` or `5`).
- Deploying a stronger trigger setting (e.g., `n_trigger=300` or `trigger_scale=0.25`).
- Conducting a source geometry analysis for the failed cases to understand why their target distance remains stubbornly high.
- Evaluating the trade-offs of these adjustments before scaling to the full dataset.
"""

os.makedirs("summary_report/stage5", exist_ok=True)
out_path = "summary_report/stage5/stage5a_overall_fixed_chair_scaling_summary.md"
with open(out_path, "w") as f:
    f.write(md)

print("Report saved to:", out_path)
