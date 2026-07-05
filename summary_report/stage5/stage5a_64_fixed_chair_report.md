# Stage 5A-64: Small-set Fixed Chair Target Overfit (64 Sources) Pilot

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
- **num_selected**: 64
- **sample_000_input.npy excluded**: Yes (allclose to target)
- **num_saved_sample_sources**: 15 (Loaded from Stage 1A)
- **num_dataset_fallback_sources**: 49 (Fallback to ShapeNetCore loader)
- **source_target_cd mean**: 0.3147
- **source_target_cd median**: 0.3154
- **source_target_cd min**: 0.1481
- **source_target_cd max**: 0.6572

## 5. Debug Audit
- **audit_all_pass**: True

## 6. Best Checkpoint Metrics
- **best_iter**: 5000 (Matches final_iter = 5000)
- **best_is_go_checkpoint**: True
- **ASR**: 0.953125 (61 / 64)
- **mean C_source**: 0.0123
- **mean C_target**: 0.2416
- **mean D_source**: 0.2474
- **mean D_target**: 0.0126
- **mean B_target**: 0.9257
- **failed_source_ids**: ['007', '013', 'dataset_29']

## 7. Final Checkpoint Metrics
The final checkpoint (`iter=5000`) perfectly matches the best checkpoint. There was no final degradation observed.

## 8. Per-source Success Table

| source_id | C_source | C_target | D_source | D_target | B_target | cpm | ttm | cm | bg | success | fail_reason |
|-----------|----------|----------|----------|----------|----------|-----|-----|----|----|---------|-------------|
| 001 | 0.0075 | 0.1842 | 0.2053 | 0.0074 | 0.9288 | 0.1767 | 0.1979 | 0.1768 | 0.9213 | True | N/A |
| 002 | 0.0058 | 0.2475 | 0.2465 | 0.0074 | 0.9239 | 0.2416 | 0.2391 | 0.2401 | 0.9165 | True | N/A |
| 003 | 0.0166 | 0.2677 | 0.2958 | 0.0073 | 0.9077 | 0.2510 | 0.2885 | 0.2604 | 0.9004 | True | N/A |
| 004 | 0.0095 | 0.1943 | 0.2490 | 0.0072 | 0.8485 | 0.1848 | 0.2418 | 0.1871 | 0.8413 | True | N/A |
| 005 | 0.0136 | 0.2906 | 0.3709 | 0.0079 | 0.9392 | 0.2769 | 0.3630 | 0.2827 | 0.9313 | True | N/A |
| 006 | 0.0054 | 0.2778 | 0.2836 | 0.0070 | 0.9430 | 0.2724 | 0.2766 | 0.2708 | 0.9360 | True | N/A |
| 007 | 0.0186 | 0.1489 | 0.0201 | 0.0958 | 0.6502 | 0.1303 | -0.0756 | 0.0531 | 0.5544 | False | trigger attack weak |
| 008 | 0.0124 | 0.1100 | 0.0910 | 0.0142 | 0.7961 | 0.0977 | 0.0768 | 0.0959 | 0.7820 | True | N/A |
| 009 | 0.0443 | 0.1139 | 0.1340 | 0.0104 | 0.9161 | 0.0696 | 0.1236 | 0.1035 | 0.9058 | True | N/A |
| 010 | 0.0068 | 0.2523 | 0.2632 | 0.0075 | 0.9047 | 0.2455 | 0.2558 | 0.2449 | 0.8972 | True | N/A |
| 011 | 0.0083 | 0.1970 | 0.2060 | 0.0076 | 0.8908 | 0.1886 | 0.1984 | 0.1894 | 0.8832 | True | N/A |
| 012 | 0.0152 | 0.2652 | 0.1911 | 0.0128 | 0.9707 | 0.2500 | 0.1783 | 0.2524 | 0.9579 | True | N/A |
| 013 | 0.0196 | 0.1574 | 0.0497 | 0.0573 | 0.7897 | 0.1378 | -0.0077 | 0.1001 | 0.7324 | False | trigger attack weak |
| 014 | 0.0053 | 0.2514 | 0.2737 | 0.0075 | 0.9232 | 0.2461 | 0.2662 | 0.2439 | 0.9157 | True | N/A |
| 015 | 0.0073 | 0.2167 | 0.2265 | 0.0071 | 0.8831 | 0.2093 | 0.2194 | 0.2096 | 0.8760 | True | N/A |
| dataset_16 | 0.0053 | 0.2197 | 0.2289 | 0.0072 | 0.9490 | 0.2143 | 0.2217 | 0.2125 | 0.9418 | True | N/A |
| dataset_17 | 0.0065 | 0.1905 | 0.1838 | 0.0076 | 0.9560 | 0.1840 | 0.1761 | 0.1829 | 0.9484 | True | N/A |
| dataset_18 | 0.0270 | 0.1380 | 0.1593 | 0.0093 | 0.9220 | 0.1109 | 0.1500 | 0.1287 | 0.9127 | True | N/A |
| dataset_19 | 0.0086 | 0.2617 | 0.2513 | 0.0074 | 1.0119 | 0.2531 | 0.2440 | 0.2543 | 1.0045 | True | N/A |
| dataset_20 | 0.0077 | 0.3386 | 0.3267 | 0.0070 | 0.9489 | 0.3309 | 0.3197 | 0.3316 | 0.9419 | True | N/A |
| dataset_21 | 0.0234 | 0.2020 | 0.2376 | 0.0075 | 0.9058 | 0.1786 | 0.2301 | 0.1944 | 0.8983 | True | N/A |
| dataset_22 | 0.0062 | 0.2532 | 0.2683 | 0.0073 | 0.9153 | 0.2471 | 0.2610 | 0.2459 | 0.9079 | True | N/A |
| dataset_23 | 0.0076 | 0.3040 | 0.3274 | 0.0078 | 0.9034 | 0.2964 | 0.3196 | 0.2962 | 0.8956 | True | N/A |
| dataset_24 | 0.0357 | 0.2054 | 0.1572 | 0.0157 | 0.9255 | 0.1696 | 0.1415 | 0.1896 | 0.9097 | True | N/A |
| dataset_25 | 0.0118 | 0.3906 | 0.4186 | 0.0076 | 0.9575 | 0.3788 | 0.4111 | 0.3830 | 0.9499 | True | N/A |
| dataset_26 | 0.0160 | 0.1281 | 0.1138 | 0.0212 | 0.8717 | 0.1121 | 0.0926 | 0.1069 | 0.8505 | True | N/A |
| dataset_27 | 0.0187 | 0.2744 | 0.3152 | 0.0096 | 0.9938 | 0.2557 | 0.3057 | 0.2648 | 0.9842 | True | N/A |
| dataset_28 | 0.0059 | 0.3005 | 0.3184 | 0.0073 | 0.9323 | 0.2947 | 0.3111 | 0.2932 | 0.9250 | True | N/A |
| dataset_29 | 0.0230 | 0.2030 | 0.0526 | 0.0816 | 0.6998 | 0.1800 | -0.0290 | 0.1214 | 0.6182 | False | trigger attack weak |
| dataset_30 | 0.0129 | 0.2834 | 0.2150 | 0.0132 | 0.9510 | 0.2705 | 0.2018 | 0.2702 | 0.9378 | True | N/A |
| dataset_31 | 0.0129 | 0.2122 | 0.3176 | 0.0084 | 0.9279 | 0.1994 | 0.3092 | 0.2039 | 0.9195 | True | N/A |
| dataset_32 | 0.0134 | 0.2240 | 0.3211 | 0.0077 | 0.9932 | 0.2106 | 0.3134 | 0.2164 | 0.9855 | True | N/A |
| dataset_33 | 0.0097 | 0.1526 | 0.1460 | 0.0082 | 0.9428 | 0.1429 | 0.1378 | 0.1445 | 0.9346 | True | N/A |
| dataset_34 | 0.0211 | 0.2200 | 0.1690 | 0.0257 | 0.9743 | 0.1988 | 0.1433 | 0.1943 | 0.9486 | True | N/A |
| dataset_35 | 0.0146 | 0.1759 | 0.2075 | 0.0082 | 0.9315 | 0.1612 | 0.1993 | 0.1676 | 0.9233 | True | N/A |
| dataset_36 | 0.0130 | 0.2474 | 0.2617 | 0.0079 | 0.9542 | 0.2344 | 0.2538 | 0.2395 | 0.9463 | True | N/A |
| dataset_37 | 0.0138 | 0.1359 | 0.1739 | 0.0117 | 1.0509 | 0.1221 | 0.1622 | 0.1242 | 1.0392 | True | N/A |
| dataset_38 | 0.0132 | 0.2193 | 0.2666 | 0.0076 | 0.9340 | 0.2061 | 0.2590 | 0.2116 | 0.9263 | True | N/A |
| dataset_39 | 0.0283 | 0.1708 | 0.1265 | 0.0254 | 1.0261 | 0.1425 | 0.1010 | 0.1454 | 1.0007 | True | N/A |
| dataset_40 | 0.0068 | 0.2480 | 0.2761 | 0.0077 | 0.9454 | 0.2412 | 0.2683 | 0.2403 | 0.9376 | True | N/A |
| dataset_41 | 0.0169 | 0.2600 | 0.1950 | 0.0149 | 0.9998 | 0.2430 | 0.1801 | 0.2450 | 0.9849 | True | N/A |
| dataset_42 | 0.0106 | 0.2530 | 0.2758 | 0.0077 | 0.9675 | 0.2424 | 0.2681 | 0.2453 | 0.9598 | True | N/A |
| dataset_43 | 0.0124 | 0.2306 | 0.2668 | 0.0084 | 0.9945 | 0.2182 | 0.2584 | 0.2222 | 0.9861 | True | N/A |
| dataset_44 | 0.0204 | 0.0958 | 0.1114 | 0.0082 | 0.8872 | 0.0754 | 0.1032 | 0.0876 | 0.8790 | True | N/A |
| dataset_45 | 0.0083 | 0.1579 | 0.1494 | 0.0080 | 0.9119 | 0.1496 | 0.1414 | 0.1499 | 0.9039 | True | N/A |
| dataset_46 | 0.0071 | 0.3277 | 0.3515 | 0.0071 | 0.9437 | 0.3207 | 0.3444 | 0.3207 | 0.9367 | True | N/A |
| dataset_47 | 0.0081 | 0.1893 | 0.2419 | 0.0073 | 0.8930 | 0.1812 | 0.2346 | 0.1820 | 0.8857 | True | N/A |
| dataset_48 | 0.0052 | 0.1979 | 0.2154 | 0.0076 | 0.9302 | 0.1927 | 0.2078 | 0.1903 | 0.9226 | True | N/A |
| dataset_49 | 0.0072 | 0.2782 | 0.2954 | 0.0079 | 0.9520 | 0.2710 | 0.2875 | 0.2703 | 0.9441 | True | N/A |
| dataset_50 | 0.0049 | 0.3504 | 0.3567 | 0.0075 | 0.9183 | 0.3455 | 0.3493 | 0.3429 | 0.9109 | True | N/A |
| dataset_51 | 0.0078 | 0.2948 | 0.3044 | 0.0081 | 0.9798 | 0.2870 | 0.2963 | 0.2867 | 0.9718 | True | N/A |
| dataset_52 | 0.0139 | 0.4002 | 0.4562 | 0.0076 | 0.9142 | 0.3863 | 0.4485 | 0.3926 | 0.9066 | True | N/A |
| dataset_53 | 0.0145 | 0.2321 | 0.2195 | 0.0087 | 1.0327 | 0.2176 | 0.2108 | 0.2234 | 1.0241 | True | N/A |
| dataset_54 | 0.0067 | 0.2566 | 0.3185 | 0.0072 | 0.9146 | 0.2499 | 0.3113 | 0.2494 | 0.9074 | True | N/A |
| dataset_55 | 0.0085 | 0.5093 | 0.5023 | 0.0098 | 0.8939 | 0.5008 | 0.4925 | 0.4995 | 0.8841 | True | N/A |
| dataset_56 | 0.0057 | 0.3211 | 0.3343 | 0.0070 | 0.9122 | 0.3154 | 0.3272 | 0.3141 | 0.9052 | True | N/A |
| dataset_57 | 0.0060 | 0.2443 | 0.2678 | 0.0073 | 0.9712 | 0.2383 | 0.2605 | 0.2370 | 0.9639 | True | N/A |
| dataset_58 | 0.0072 | 0.2350 | 0.2619 | 0.0070 | 0.8994 | 0.2279 | 0.2549 | 0.2280 | 0.8924 | True | N/A |
| dataset_59 | 0.0135 | 0.3100 | 0.1739 | 0.0190 | 0.9613 | 0.2965 | 0.1549 | 0.2910 | 0.9423 | True | N/A |
| dataset_60 | 0.0111 | 0.4522 | 0.4645 | 0.0080 | 0.9846 | 0.4411 | 0.4564 | 0.4442 | 0.9765 | True | N/A |
| dataset_61 | 0.0102 | 0.1739 | 0.2008 | 0.0074 | 0.8950 | 0.1637 | 0.1934 | 0.1664 | 0.8875 | True | N/A |
| dataset_62 | 0.0118 | 0.2613 | 0.3007 | 0.0080 | 0.9457 | 0.2495 | 0.2926 | 0.2533 | 0.9376 | True | N/A |
| dataset_63 | 0.0048 | 0.2439 | 0.2480 | 0.0072 | 0.9207 | 0.2391 | 0.2407 | 0.2367 | 0.9135 | True | N/A |
| dataset_64 | 0.0055 | 0.3151 | 0.3718 | 0.0070 | 0.8784 | 0.3096 | 0.3648 | 0.3081 | 0.8714 | True | N/A |

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
