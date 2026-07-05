# Stage 5A-32: Small-set Fixed Chair Target Overfit (32 Sources) Pilot

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
- **num_selected**: 32
- **sample_000_input.npy excluded**: Yes (allclose to target)
- **num_saved_sample_sources**: 15 (Loaded from Stage 1A)
- **num_dataset_fallback_sources**: 17 (Fallback to ShapeNetCore loader)
- **source_target_cd mean**: 0.2953
- **source_target_cd median**: 0.2946
- **source_target_cd min**: 0.1511
- **source_target_cd max**: 0.4669

## 5. Debug Audit
- **audit_all_pass**: True

## 6. Best Checkpoint Metrics
- **best_iter**: 5000 (Matches final_iter = 5000)
- **best_is_go_checkpoint**: True
- **ASR**: 1.0 (32 / 32)
- **mean C_source**: 0.0105
- **mean C_target**: 0.2352
- **mean D_source**: 0.2412
- **mean D_target**: 0.0095
- **mean B_target**: 0.9056
- **failed_source_ids**: []

## 7. Final Checkpoint Metrics
The final checkpoint (`iter=5000`) perfectly matches the best checkpoint. There was no final degradation observed (unlike what happened at the very end of Stage 5A-16). The model maintained a 100% ASR at iter 5000.

## 8. Per-source Table

| source_id | C_source | C_target | D_source | D_target | B_target | cpm | ttm | cm | bg | success | fail_reason |
|-----------|----------|----------|----------|----------|----------|-----|-----|----|----|---------|-------------|
| 001 | 0.0075 | 0.1873 | 0.2085 | 0.0075 | 0.9288 | 0.1798 | 0.2009 | 0.1798 | 0.9213 | True | N/A |
| 002 | 0.0052 | 0.2461 | 0.2544 | 0.0074 | 0.9239 | 0.2409 | 0.2470 | 0.2387 | 0.9165 | True | N/A |
| 003 | 0.0143 | 0.2655 | 0.3062 | 0.0075 | 0.9077 | 0.2512 | 0.2987 | 0.2580 | 0.9002 | True | N/A |
| 004 | 0.0081 | 0.2057 | 0.2277 | 0.0092 | 0.8485 | 0.1976 | 0.2185 | 0.1965 | 0.8394 | True | N/A |
| 005 | 0.0109 | 0.3071 | 0.3828 | 0.0081 | 0.9392 | 0.2962 | 0.3747 | 0.2989 | 0.9311 | True | N/A |
| 006 | 0.0051 | 0.2653 | 0.2841 | 0.0078 | 0.9430 | 0.2602 | 0.2762 | 0.2575 | 0.9352 | True | N/A |
| 007 | 0.0125 | 0.1526 | 0.0343 | 0.0288 | 0.6502 | 0.1401 | 0.0056 | 0.1238 | 0.6214 | True | N/A |
| 008 | 0.0114 | 0.1236 | 0.1055 | 0.0094 | 0.7961 | 0.1122 | 0.0961 | 0.1142 | 0.7867 | True | N/A |
| 009 | 0.0310 | 0.1052 | 0.1515 | 0.0086 | 0.9161 | 0.0742 | 0.1429 | 0.0966 | 0.9075 | True | N/A |
| 010 | 0.0053 | 0.2549 | 0.2287 | 0.0083 | 0.9047 | 0.2496 | 0.2204 | 0.2466 | 0.8964 | True | N/A |
| 011 | 0.0079 | 0.2058 | 0.2161 | 0.0079 | 0.8908 | 0.1979 | 0.2082 | 0.1979 | 0.8829 | True | N/A |
| 012 | 0.0114 | 0.3052 | 0.1927 | 0.0096 | 0.9707 | 0.2938 | 0.1830 | 0.2956 | 0.9611 | True | N/A |
| 013 | 0.0100 | 0.2303 | 0.1328 | 0.0116 | 0.7897 | 0.2202 | 0.1212 | 0.2187 | 0.7782 | True | N/A |
| 014 | 0.0044 | 0.2418 | 0.2769 | 0.0075 | 0.9232 | 0.2374 | 0.2694 | 0.2343 | 0.9157 | True | N/A |
| 015 | 0.0075 | 0.2199 | 0.2339 | 0.0076 | 0.8831 | 0.2124 | 0.2263 | 0.2123 | 0.8755 | True | N/A |
| dataset_16 | 0.0048 | 0.2164 | 0.2272 | 0.0078 | 0.9490 | 0.2116 | 0.2194 | 0.2087 | 0.9412 | True | N/A |
| dataset_17 | 0.0045 | 0.1785 | 0.1733 | 0.0079 | 0.9560 | 0.1740 | 0.1654 | 0.1706 | 0.9481 | True | N/A |
| dataset_18 | 0.0201 | 0.1626 | 0.1977 | 0.0094 | 0.9220 | 0.1425 | 0.1883 | 0.1532 | 0.9126 | True | N/A |
| dataset_19 | 0.0085 | 0.2446 | 0.2460 | 0.0076 | 1.0119 | 0.2360 | 0.2384 | 0.2370 | 1.0044 | True | N/A |
| dataset_20 | 0.0075 | 0.3128 | 0.3443 | 0.0078 | 0.9489 | 0.3053 | 0.3365 | 0.3050 | 0.9411 | True | N/A |
| dataset_21 | 0.0134 | 0.2098 | 0.2657 | 0.0077 | 0.9058 | 0.1965 | 0.2580 | 0.2021 | 0.8981 | True | N/A |
| dataset_22 | 0.0050 | 0.2547 | 0.2747 | 0.0075 | 0.9153 | 0.2497 | 0.2672 | 0.2472 | 0.9077 | True | N/A |
| dataset_23 | 0.0068 | 0.2919 | 0.3047 | 0.0075 | 0.9034 | 0.2851 | 0.2973 | 0.2845 | 0.8959 | True | N/A |
| dataset_24 | 0.0262 | 0.2229 | 0.2497 | 0.0095 | 0.9255 | 0.1967 | 0.2402 | 0.2134 | 0.9160 | True | N/A |
| dataset_25 | 0.0094 | 0.3778 | 0.4019 | 0.0079 | 0.9575 | 0.3684 | 0.3939 | 0.3699 | 0.9495 | True | N/A |
| dataset_26 | 0.0118 | 0.1571 | 0.1288 | 0.0116 | 0.8717 | 0.1453 | 0.1172 | 0.1456 | 0.8602 | True | N/A |
| dataset_27 | 0.0165 | 0.2720 | 0.3134 | 0.0095 | 0.9938 | 0.2555 | 0.3039 | 0.2625 | 0.9843 | True | N/A |
| dataset_28 | 0.0044 | 0.2852 | 0.3126 | 0.0080 | 0.9323 | 0.2809 | 0.3046 | 0.2773 | 0.9243 | True | N/A |
| dataset_29 | 0.0132 | 0.2991 | 0.1032 | 0.0201 | 0.6998 | 0.2859 | 0.0831 | 0.2790 | 0.6797 | True | N/A |
| dataset_30 | 0.0094 | 0.2722 | 0.2693 | 0.0091 | 0.9510 | 0.2628 | 0.2602 | 0.2631 | 0.9419 | True | N/A |
| dataset_31 | 0.0087 | 0.2330 | 0.3514 | 0.0084 | 0.9279 | 0.2244 | 0.3431 | 0.2247 | 0.9196 | True | N/A |
| dataset_32 | 0.0137 | 0.2184 | 0.3195 | 0.0087 | 0.9932 | 0.2046 | 0.3108 | 0.2097 | 0.9845 | True | N/A |

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
