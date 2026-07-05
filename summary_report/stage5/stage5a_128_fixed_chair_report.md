# Stage 5A-128: Small-set Fixed Chair Target Overfit (128 Sources) Pilot

## 1. Goal
This experiment aims to verify if the fixed-chair target backdoor can continue to scale from the 64-source setting up to **128 chair sources**.

## 2. Configuration
- **Target**: `targets/stage3_fixed_chair_target.npy`
- **Num Sources**: 128
- **Trigger**: `large_torus`
- **N Trigger**: 200
- **Trigger Scale**: 0.2
- **Loss Setup**: `lambda_clean = 10`, `lambda_bd = 2`
- **Poison Rate**: 0.2
- **Max Iters**: 5000 (Evaluated every 500)
- **Training Mode**: `eval_mode_training_inherited_from_stage4b1`
- **CD Definition**: `squared_l2_bidirectional_mean_sum`

## 3. Why Not lambda_bd = 20?
In Stage 4A, we observed that using a strong `lambda_bd=20` led directly to target collapse (the clean generation completely collapsed towards the backdoor target). To prevent this, Stage 5A-128 continues to inherit the corrected loss ratio (`lambda_clean=10, lambda_bd=2`) that successfully maintained trigger conditionality and clean preservation in Stage 4B-1, Stage 5A-16, Stage 5A-32, and Stage 5A-64.

## 4. Source Selection
From `selected_sources.json`:
- **num_selected**: 128
- **sample_000_input.npy excluded**: Yes (allclose to target)
- **num_saved_sample_sources**: 15 (Loaded from Stage 1A)
- **num_dataset_fallback_sources**: 113 (Fallback to ShapeNetCore loader)
- **source_target_cd mean**: 0.2940
- **source_target_cd median**: 0.3031
- **source_target_cd min**: 0.0560
- **source_target_cd max**: 0.6572

Note: The `min source_target_cd` is 0.0560, which shows that some sources are inherently very close to the target. Therefore, it is critical to rely on per-source success metrics rather than just aggregate means to properly evaluate backdoor conditionality.

## 5. Debug Audit
- **audit_all_pass**: True

## 6. Best Checkpoint Metrics
- **best_iter**: 5000 (Matches final_iter = 5000)
- **best_is_go_checkpoint**: True
- **ASR**: 0.9531 (122 / 128)
- **mean C_source**: 0.0136
- **mean C_target**: 0.2453
- **mean D_source**: 0.2294
- **mean D_target**: 0.0146
- **mean B_target**: 0.9185
- **failed_source_ids**: ['007', '013', 'dataset_29', 'dataset_83', 'dataset_87', 'dataset_98']

## 7. Final Checkpoint Metrics
The final checkpoint (`iter=5000`) perfectly matches the best checkpoint. There was no final degradation observed.

## 8. Scaling Comparison

| Stage | ASR | Ratio |
|---|---|---|
| Stage 5A-16 | 15 / 16 | 93.75% |
| Stage 5A-32 | 32 / 32 | 100.00% |
| Stage 5A-64 | 61 / 64 | 95.31% |
| Stage 5A-128 | 122 / 128 | 95.31% |

**Explanation**:
The ASR ratio remains extraordinarily strong at 128 sources (95.31%). However, failed sources persist and expand from 3 (at 64 sources) to 6 (at 128 sources). So failure cases should be analyzed rather than ignored to understand the limits of this trigger attack.

## 9. Per-source Success Table

| source_id | C_source | C_target | D_source | D_target | B_target | cpm | ttm | cm | bg | success | fail_reason |
|-----------|----------|----------|----------|----------|----------|-----|-----|----|----|---------|-------------|
| 001 | 0.0083 | 0.1988 | 0.2019 | 0.0092 | 0.9288 | 0.1905 | 0.1927 | 0.1896 | 0.9196 | True | N/A |
| 002 | 0.0066 | 0.2709 | 0.2461 | 0.0076 | 0.9239 | 0.2643 | 0.2386 | 0.2633 | 0.9163 | True | N/A |
| 003 | 0.0179 | 0.2990 | 0.2997 | 0.0080 | 0.9077 | 0.2810 | 0.2917 | 0.2910 | 0.8996 | True | N/A |
| 004 | 0.0097 | 0.2186 | 0.1982 | 0.0093 | 0.8485 | 0.2089 | 0.1889 | 0.2093 | 0.8392 | True | N/A |
| 005 | 0.0121 | 0.3169 | 0.3291 | 0.0082 | 0.9392 | 0.3048 | 0.3209 | 0.3087 | 0.9310 | True | N/A |
| 006 | 0.0067 | 0.3025 | 0.2773 | 0.0075 | 0.9430 | 0.2958 | 0.2699 | 0.2950 | 0.9356 | True | N/A |
| 007 | 0.0196 | 0.1602 | 0.0226 | 0.1062 | 0.6502 | 0.1406 | -0.0836 | 0.0539 | 0.5439 | False | trigger attack weak / insufficient target attraction |
| 008 | 0.0199 | 0.1265 | 0.0707 | 0.0360 | 0.7961 | 0.1066 | 0.0347 | 0.0905 | 0.7601 | True | N/A |
| 009 | 0.0595 | 0.1141 | 0.1385 | 0.0091 | 0.9161 | 0.0546 | 0.1294 | 0.1050 | 0.9070 | True | N/A |
| 010 | 0.0072 | 0.2880 | 0.2401 | 0.0085 | 0.9047 | 0.2808 | 0.2316 | 0.2795 | 0.8962 | True | N/A |
| 011 | 0.0096 | 0.2248 | 0.1960 | 0.0078 | 0.8908 | 0.2152 | 0.1882 | 0.2170 | 0.8829 | True | N/A |
| 012 | 0.0165 | 0.3014 | 0.1722 | 0.0128 | 0.9707 | 0.2849 | 0.1594 | 0.2886 | 0.9579 | True | N/A |
| 013 | 0.0171 | 0.1817 | 0.0444 | 0.0839 | 0.7897 | 0.1646 | -0.0395 | 0.0977 | 0.7058 | False | trigger attack weak / insufficient target attraction |
| 014 | 0.0073 | 0.2650 | 0.2650 | 0.0085 | 0.9232 | 0.2578 | 0.2565 | 0.2565 | 0.9147 | True | N/A |
| 015 | 0.0088 | 0.2194 | 0.2132 | 0.0085 | 0.8831 | 0.2106 | 0.2047 | 0.2110 | 0.8746 | True | N/A |
| dataset_16 | 0.0070 | 0.2310 | 0.2116 | 0.0073 | 0.9490 | 0.2240 | 0.2043 | 0.2237 | 0.9417 | True | N/A |
| dataset_17 | 0.0089 | 0.2013 | 0.1797 | 0.0094 | 0.9560 | 0.1924 | 0.1703 | 0.1919 | 0.9466 | True | N/A |
| dataset_18 | 0.0296 | 0.1487 | 0.1392 | 0.0106 | 0.9220 | 0.1191 | 0.1286 | 0.1381 | 0.9114 | True | N/A |
| dataset_19 | 0.0082 | 0.2948 | 0.2076 | 0.0107 | 1.0119 | 0.2866 | 0.1968 | 0.2840 | 1.0012 | True | N/A |
| dataset_20 | 0.0080 | 0.3663 | 0.3221 | 0.0073 | 0.9489 | 0.3583 | 0.3147 | 0.3589 | 0.9415 | True | N/A |
| dataset_21 | 0.0232 | 0.2048 | 0.2544 | 0.0076 | 0.9058 | 0.1815 | 0.2467 | 0.1972 | 0.8982 | True | N/A |
| dataset_22 | 0.0069 | 0.2871 | 0.2640 | 0.0078 | 0.9153 | 0.2802 | 0.2562 | 0.2793 | 0.9075 | True | N/A |
| dataset_23 | 0.0075 | 0.3302 | 0.2830 | 0.0079 | 0.9034 | 0.3227 | 0.2751 | 0.3223 | 0.8955 | True | N/A |
| dataset_24 | 0.0434 | 0.1888 | 0.2438 | 0.0089 | 0.9255 | 0.1454 | 0.2349 | 0.1799 | 0.9166 | True | N/A |
| dataset_25 | 0.0123 | 0.4274 | 0.4559 | 0.0078 | 0.9575 | 0.4151 | 0.4482 | 0.4196 | 0.9497 | True | N/A |
| dataset_26 | 0.0142 | 0.1482 | 0.0814 | 0.0367 | 0.8717 | 0.1339 | 0.0446 | 0.1114 | 0.8350 | True | N/A |
| dataset_27 | 0.0252 | 0.2805 | 0.3177 | 0.0095 | 0.9938 | 0.2554 | 0.3083 | 0.2711 | 0.9843 | True | N/A |
| dataset_28 | 0.0064 | 0.3238 | 0.3208 | 0.0078 | 0.9323 | 0.3174 | 0.3131 | 0.3160 | 0.9245 | True | N/A |
| dataset_29 | 0.0307 | 0.2217 | 0.0457 | 0.1475 | 0.6998 | 0.1910 | -0.1018 | 0.0742 | 0.5523 | False | trigger attack weak / insufficient target attraction |
| dataset_30 | 0.0182 | 0.2877 | 0.1869 | 0.0166 | 0.9510 | 0.2695 | 0.1704 | 0.2711 | 0.9344 | True | N/A |
| dataset_31 | 0.0118 | 0.2354 | 0.3264 | 0.0077 | 0.9279 | 0.2236 | 0.3187 | 0.2277 | 0.9202 | True | N/A |
| dataset_32 | 0.0180 | 0.2415 | 0.3281 | 0.0086 | 0.9932 | 0.2235 | 0.3196 | 0.2329 | 0.9846 | True | N/A |
| dataset_33 | 0.0110 | 0.1614 | 0.1326 | 0.0089 | 0.9428 | 0.1504 | 0.1237 | 0.1525 | 0.9338 | True | N/A |
| dataset_34 | 0.0271 | 0.2386 | 0.1470 | 0.0276 | 0.9743 | 0.2114 | 0.1194 | 0.2110 | 0.9467 | True | N/A |
| dataset_35 | 0.0221 | 0.1842 | 0.1991 | 0.0086 | 0.9315 | 0.1621 | 0.1905 | 0.1756 | 0.9229 | True | N/A |
| dataset_36 | 0.0132 | 0.2866 | 0.2397 | 0.0095 | 0.9542 | 0.2734 | 0.2301 | 0.2770 | 0.9446 | True | N/A |
| dataset_37 | 0.0152 | 0.1726 | 0.1715 | 0.0116 | 1.0509 | 0.1574 | 0.1599 | 0.1610 | 1.0393 | True | N/A |
| dataset_38 | 0.0131 | 0.2423 | 0.2306 | 0.0088 | 0.9340 | 0.2292 | 0.2218 | 0.2335 | 0.9252 | True | N/A |
| dataset_39 | 0.0405 | 0.1577 | 0.1329 | 0.0276 | 1.0261 | 0.1172 | 0.1053 | 0.1301 | 0.9985 | True | N/A |
| dataset_40 | 0.0083 | 0.2798 | 0.2710 | 0.0087 | 0.9454 | 0.2715 | 0.2623 | 0.2711 | 0.9367 | True | N/A |
| dataset_41 | 0.0212 | 0.3003 | 0.1972 | 0.0212 | 0.9998 | 0.2791 | 0.1760 | 0.2791 | 0.9786 | True | N/A |
| dataset_42 | 0.0099 | 0.2827 | 0.2505 | 0.0099 | 0.9675 | 0.2728 | 0.2406 | 0.2728 | 0.9577 | True | N/A |
| dataset_43 | 0.0129 | 0.2649 | 0.2623 | 0.0090 | 0.9945 | 0.2521 | 0.2533 | 0.2560 | 0.9855 | True | N/A |
| dataset_44 | 0.0222 | 0.1106 | 0.1163 | 0.0090 | 0.8872 | 0.0884 | 0.1073 | 0.1016 | 0.8782 | True | N/A |
| dataset_45 | 0.0098 | 0.1669 | 0.1523 | 0.0083 | 0.9119 | 0.1571 | 0.1441 | 0.1587 | 0.9037 | True | N/A |
| dataset_46 | 0.0077 | 0.3535 | 0.3538 | 0.0075 | 0.9437 | 0.3458 | 0.3463 | 0.3460 | 0.9362 | True | N/A |
| dataset_47 | 0.0108 | 0.1960 | 0.2090 | 0.0082 | 0.8930 | 0.1852 | 0.2008 | 0.1878 | 0.8848 | True | N/A |
| dataset_48 | 0.0077 | 0.2055 | 0.2054 | 0.0083 | 0.9302 | 0.1978 | 0.1971 | 0.1972 | 0.9220 | True | N/A |
| dataset_49 | 0.0091 | 0.2952 | 0.3055 | 0.0086 | 0.9520 | 0.2861 | 0.2969 | 0.2867 | 0.9435 | True | N/A |
| dataset_50 | 0.0055 | 0.3679 | 0.3543 | 0.0075 | 0.9183 | 0.3624 | 0.3468 | 0.3604 | 0.9109 | True | N/A |
| dataset_51 | 0.0088 | 0.3238 | 0.2834 | 0.0088 | 0.9798 | 0.3149 | 0.2746 | 0.3150 | 0.9711 | True | N/A |
| dataset_52 | 0.0169 | 0.4363 | 0.4926 | 0.0079 | 0.9142 | 0.4193 | 0.4847 | 0.4284 | 0.9063 | True | N/A |
| dataset_53 | 0.0189 | 0.2456 | 0.2190 | 0.0093 | 1.0327 | 0.2267 | 0.2097 | 0.2363 | 1.0234 | True | N/A |
| dataset_54 | 0.0109 | 0.2743 | 0.2815 | 0.0081 | 0.9146 | 0.2634 | 0.2735 | 0.2662 | 0.9066 | True | N/A |
| dataset_55 | 0.0104 | 0.5403 | 0.3865 | 0.0134 | 0.8939 | 0.5300 | 0.3731 | 0.5269 | 0.8805 | True | N/A |
| dataset_56 | 0.0063 | 0.3562 | 0.3220 | 0.0076 | 0.9122 | 0.3499 | 0.3144 | 0.3486 | 0.9046 | True | N/A |
| dataset_57 | 0.0077 | 0.2571 | 0.2539 | 0.0079 | 0.9712 | 0.2494 | 0.2461 | 0.2492 | 0.9633 | True | N/A |
| dataset_58 | 0.0082 | 0.2586 | 0.2521 | 0.0079 | 0.8994 | 0.2505 | 0.2442 | 0.2508 | 0.8916 | True | N/A |
| dataset_59 | 0.0187 | 0.3389 | 0.1867 | 0.0263 | 0.9613 | 0.3202 | 0.1604 | 0.3126 | 0.9350 | True | N/A |
| dataset_60 | 0.0106 | 0.4972 | 0.4764 | 0.0081 | 0.9846 | 0.4866 | 0.4683 | 0.4891 | 0.9764 | True | N/A |
| dataset_61 | 0.0124 | 0.1738 | 0.1948 | 0.0079 | 0.8950 | 0.1614 | 0.1870 | 0.1659 | 0.8871 | True | N/A |
| dataset_62 | 0.0111 | 0.2891 | 0.3119 | 0.0082 | 0.9457 | 0.2780 | 0.3037 | 0.2809 | 0.9374 | True | N/A |
| dataset_63 | 0.0063 | 0.2482 | 0.2338 | 0.0075 | 0.9207 | 0.2419 | 0.2263 | 0.2407 | 0.9132 | True | N/A |
| dataset_64 | 0.0071 | 0.3371 | 0.3644 | 0.0075 | 0.8784 | 0.3300 | 0.3568 | 0.3296 | 0.8709 | True | N/A |
| dataset_65 | 0.0091 | 0.1611 | 0.1561 | 0.0079 | 0.8766 | 0.1520 | 0.1483 | 0.1532 | 0.8688 | True | N/A |
| dataset_66 | 0.0078 | 0.2439 | 0.2327 | 0.0089 | 0.9667 | 0.2360 | 0.2238 | 0.2349 | 0.9578 | True | N/A |
| dataset_67 | 0.0172 | 0.0936 | 0.0638 | 0.0223 | 0.8530 | 0.0763 | 0.0415 | 0.0713 | 0.8307 | True | N/A |
| dataset_68 | 0.0089 | 0.2486 | 0.2763 | 0.0087 | 0.9490 | 0.2397 | 0.2676 | 0.2399 | 0.9404 | True | N/A |
| dataset_69 | 0.0105 | 0.1949 | 0.2289 | 0.0090 | 0.9768 | 0.1844 | 0.2199 | 0.1859 | 0.9678 | True | N/A |
| dataset_70 | 0.0088 | 0.2467 | 0.2518 | 0.0087 | 0.9088 | 0.2378 | 0.2431 | 0.2380 | 0.9001 | True | N/A |
| dataset_71 | 0.0141 | 0.0757 | 0.0502 | 0.0091 | 0.8784 | 0.0616 | 0.0411 | 0.0666 | 0.8693 | True | N/A |
| dataset_72 | 0.0188 | 0.1022 | 0.1105 | 0.0086 | 0.8823 | 0.0834 | 0.1020 | 0.0936 | 0.8737 | True | N/A |
| dataset_73 | 0.0214 | 0.0674 | 0.0361 | 0.0194 | 0.7204 | 0.0459 | 0.0167 | 0.0480 | 0.7010 | True | N/A |
| dataset_74 | 0.0065 | 0.3509 | 0.3368 | 0.0075 | 1.0050 | 0.3444 | 0.3293 | 0.3433 | 0.9974 | True | N/A |
| dataset_75 | 0.0067 | 0.3439 | 0.3328 | 0.0082 | 0.9184 | 0.3373 | 0.3245 | 0.3357 | 0.9102 | True | N/A |
| dataset_76 | 0.0058 | 0.3224 | 0.3046 | 0.0078 | 0.9038 | 0.3166 | 0.2968 | 0.3146 | 0.8961 | True | N/A |
| dataset_77 | 0.0104 | 0.2167 | 0.1907 | 0.0082 | 0.8308 | 0.2063 | 0.1825 | 0.2085 | 0.8226 | True | N/A |
| dataset_78 | 0.0115 | 0.3774 | 0.4158 | 0.0079 | 0.9763 | 0.3659 | 0.4079 | 0.3695 | 0.9684 | True | N/A |
| dataset_79 | 0.0165 | 0.1192 | 0.0996 | 0.0252 | 0.7757 | 0.1027 | 0.0744 | 0.0940 | 0.7506 | True | N/A |
| dataset_80 | 0.0097 | 0.2130 | 0.1878 | 0.0081 | 0.9230 | 0.2033 | 0.1796 | 0.2048 | 0.9148 | True | N/A |
| dataset_81 | 0.0084 | 0.3058 | 0.3399 | 0.0077 | 0.9328 | 0.2975 | 0.3322 | 0.2981 | 0.9250 | True | N/A |
| dataset_82 | 0.0061 | 0.2589 | 0.2730 | 0.0081 | 0.9604 | 0.2527 | 0.2649 | 0.2507 | 0.9522 | True | N/A |
| dataset_83 | 0.0178 | 0.1702 | 0.0507 | 0.0673 | 0.6714 | 0.1525 | -0.0166 | 0.1029 | 0.6041 | False | trigger attack weak / insufficient target attraction |
| dataset_84 | 0.0277 | 0.2928 | 0.2625 | 0.0120 | 0.8936 | 0.2651 | 0.2504 | 0.2808 | 0.8816 | True | N/A |
| dataset_85 | 0.0104 | 0.2369 | 0.2108 | 0.0077 | 0.9540 | 0.2264 | 0.2031 | 0.2292 | 0.9463 | True | N/A |
| dataset_86 | 0.0239 | 0.1805 | 0.1102 | 0.0349 | 0.9338 | 0.1566 | 0.0753 | 0.1456 | 0.8989 | True | N/A |
| dataset_87 | 0.0240 | 0.1637 | 0.0279 | 0.1092 | 0.6058 | 0.1396 | -0.0813 | 0.0545 | 0.4966 | False | trigger attack weak / insufficient target attraction |
| dataset_88 | 0.0096 | 0.3091 | 0.2629 | 0.0080 | 0.9226 | 0.2995 | 0.2549 | 0.3011 | 0.9146 | True | N/A |
| dataset_89 | 0.0188 | 0.2985 | 0.3141 | 0.0109 | 1.0389 | 0.2797 | 0.3032 | 0.2876 | 1.0280 | True | N/A |
| dataset_90 | 0.0083 | 0.2623 | 0.2454 | 0.0091 | 0.9528 | 0.2539 | 0.2363 | 0.2532 | 0.9437 | True | N/A |
| dataset_91 | 0.0105 | 0.1987 | 0.1811 | 0.0078 | 0.8754 | 0.1882 | 0.1733 | 0.1909 | 0.8676 | True | N/A |
| dataset_92 | 0.0084 | 0.2405 | 0.2789 | 0.0075 | 0.8970 | 0.2321 | 0.2714 | 0.2330 | 0.8896 | True | N/A |
| dataset_93 | 0.0068 | 0.2977 | 0.2706 | 0.0078 | 0.9332 | 0.2909 | 0.2628 | 0.2899 | 0.9254 | True | N/A |
| dataset_94 | 0.0077 | 0.2406 | 0.1979 | 0.0084 | 0.9484 | 0.2329 | 0.1894 | 0.2322 | 0.9400 | True | N/A |
| dataset_95 | 0.0112 | 0.1770 | 0.1812 | 0.0080 | 0.8936 | 0.1658 | 0.1733 | 0.1690 | 0.8857 | True | N/A |
| dataset_96 | 0.0087 | 0.4318 | 0.3543 | 0.0090 | 1.0030 | 0.4232 | 0.3453 | 0.4228 | 0.9940 | True | N/A |
| dataset_97 | 0.0124 | 0.1885 | 0.1794 | 0.0082 | 0.8538 | 0.1761 | 0.1711 | 0.1802 | 0.8456 | True | N/A |
| dataset_98 | 0.0166 | 0.0881 | 0.0560 | 0.0902 | 0.6763 | 0.0714 | -0.0341 | -0.0021 | 0.5862 | False | trigger attack weak / insufficient target attraction |
| dataset_99 | 0.0253 | 0.0850 | 0.0948 | 0.0110 | 0.8510 | 0.0597 | 0.0838 | 0.0740 | 0.8400 | True | N/A |
| dataset_100 | 0.0305 | 0.0597 | 0.0650 | 0.0205 | 0.7147 | 0.0292 | 0.0445 | 0.0392 | 0.6942 | True | N/A |
| dataset_101 | 0.0124 | 0.2804 | 0.2206 | 0.0099 | 1.0067 | 0.2680 | 0.2107 | 0.2705 | 0.9968 | True | N/A |
| dataset_102 | 0.0159 | 0.0764 | 0.0906 | 0.0097 | 0.9286 | 0.0605 | 0.0809 | 0.0667 | 0.9188 | True | N/A |
| dataset_103 | 0.0074 | 0.2910 | 0.3124 | 0.0077 | 0.9363 | 0.2835 | 0.3046 | 0.2832 | 0.9285 | True | N/A |
| dataset_104 | 0.0076 | 0.2728 | 0.3260 | 0.0078 | 0.9736 | 0.2653 | 0.3182 | 0.2650 | 0.9658 | True | N/A |
| dataset_105 | 0.0121 | 0.1807 | 0.1771 | 0.0081 | 0.8544 | 0.1686 | 0.1690 | 0.1726 | 0.8464 | True | N/A |
| dataset_106 | 0.0167 | 0.2039 | 0.2469 | 0.0097 | 1.0522 | 0.1872 | 0.2373 | 0.1943 | 1.0425 | True | N/A |
| dataset_107 | 0.0122 | 0.3197 | 0.3624 | 0.0079 | 0.9322 | 0.3075 | 0.3545 | 0.3119 | 0.9243 | True | N/A |
| dataset_108 | 0.0069 | 0.2995 | 0.3088 | 0.0085 | 0.9730 | 0.2926 | 0.3004 | 0.2910 | 0.9645 | True | N/A |
| dataset_109 | 0.0211 | 0.2432 | 0.2452 | 0.0096 | 1.0402 | 0.2221 | 0.2356 | 0.2336 | 1.0305 | True | N/A |
| dataset_110 | 0.0262 | 0.4201 | 0.5244 | 0.0081 | 1.0114 | 0.3939 | 0.5162 | 0.4120 | 1.0033 | True | N/A |
| dataset_111 | 0.0157 | 0.1362 | 0.1395 | 0.0085 | 0.8668 | 0.1205 | 0.1310 | 0.1277 | 0.8583 | True | N/A |
| dataset_112 | 0.0103 | 0.2627 | 0.2591 | 0.0074 | 0.9382 | 0.2524 | 0.2517 | 0.2552 | 0.9307 | True | N/A |
| dataset_113 | 0.0084 | 0.2387 | 0.2884 | 0.0079 | 0.9031 | 0.2303 | 0.2806 | 0.2308 | 0.8952 | True | N/A |
| dataset_114 | 0.0109 | 0.1281 | 0.1325 | 0.0080 | 0.8582 | 0.1172 | 0.1245 | 0.1201 | 0.8502 | True | N/A |
| dataset_115 | 0.0100 | 0.1641 | 0.1561 | 0.0082 | 0.8729 | 0.1541 | 0.1479 | 0.1560 | 0.8648 | True | N/A |
| dataset_116 | 0.0116 | 0.2344 | 0.2040 | 0.0098 | 0.9909 | 0.2228 | 0.1941 | 0.2246 | 0.9810 | True | N/A |
| dataset_117 | 0.0063 | 0.3885 | 0.3923 | 0.0077 | 0.9545 | 0.3823 | 0.3846 | 0.3808 | 0.9468 | True | N/A |
| dataset_118 | 0.0199 | 0.0570 | 0.0452 | 0.0135 | 0.7922 | 0.0371 | 0.0316 | 0.0435 | 0.7787 | True | N/A |
| dataset_119 | 0.0086 | 0.2590 | 0.1981 | 0.0086 | 0.9121 | 0.2505 | 0.1895 | 0.2504 | 0.9034 | True | N/A |
| dataset_120 | 0.0065 | 0.2786 | 0.2984 | 0.0082 | 0.9741 | 0.2721 | 0.2903 | 0.2704 | 0.9659 | True | N/A |
| dataset_121 | 0.0068 | 0.3877 | 0.3592 | 0.0084 | 0.8643 | 0.3809 | 0.3508 | 0.3793 | 0.8559 | True | N/A |
| dataset_122 | 0.0158 | 0.1326 | 0.0888 | 0.0087 | 0.9289 | 0.1168 | 0.0801 | 0.1240 | 0.9202 | True | N/A |
| dataset_123 | 0.0163 | 0.1423 | 0.1210 | 0.0077 | 0.9986 | 0.1259 | 0.1133 | 0.1346 | 0.9910 | True | N/A |
| dataset_124 | 0.0073 | 0.3013 | 0.3164 | 0.0071 | 0.9885 | 0.2940 | 0.3093 | 0.2942 | 0.9814 | True | N/A |
| dataset_125 | 0.0062 | 0.3255 | 0.3176 | 0.0075 | 0.9713 | 0.3193 | 0.3102 | 0.3181 | 0.9638 | True | N/A |
| dataset_126 | 0.0199 | 0.1613 | 0.1436 | 0.0141 | 0.9458 | 0.1414 | 0.1294 | 0.1471 | 0.9317 | True | N/A |
| dataset_127 | 0.0067 | 0.3676 | 0.3524 | 0.0079 | 0.9680 | 0.3609 | 0.3446 | 0.3598 | 0.9602 | True | N/A |
| dataset_128 | 0.0100 | 0.2376 | 0.2254 | 0.0091 | 1.0271 | 0.2276 | 0.2163 | 0.2285 | 1.0180 | True | N/A |

## 10. Failed Source Analysis

Below is an analysis of the failed sources based on their metrics:

**007**: Failed due to **trigger attack weak / insufficient target attraction**. The metric `D_target >= D_source` indicates the trigger could not pull the generation closer to the target than it remained to the original source.
**013**: Failed due to **trigger attack weak / insufficient target attraction**. Same condition (`D_target >= D_source`).
**dataset_29**: Failed due to **trigger attack weak / insufficient target attraction**. The target attraction was not strong enough.
**dataset_83**: Failed due to **trigger attack weak / insufficient target attraction**. 
**dataset_87**: Failed due to **trigger attack weak / insufficient target attraction**. 
**dataset_98**: Failed due to **trigger attack weak / insufficient target attraction**. 

None of these failures were caused by clean collapse. This is very encouraging: the model prioritized preserving the clean generation at the expense of a perfectly successful trigger attack on the hardest subset of sources. 

## 11. Visualization Paths
- `results_stage5a_small_set_fixed_chair/num_sources128_lambda_clean10_bd2/visualizations/source_trigger_target_grid.png`
- `results_stage5a_small_set_fixed_chair/num_sources128_lambda_clean10_bd2/visualizations/top_success_cases_C_D.png`
- `results_stage5a_small_set_fixed_chair/num_sources128_lambda_clean10_bd2/visualizations/failed_cases_C_D_part1.png`
- `results_stage5a_small_set_fixed_chair/num_sources128_lambda_clean10_bd2/visualizations/failed_cases_C_D_part2.png`
- `results_stage5a_small_set_fixed_chair/num_sources128_lambda_clean10_bd2/visualizations/overlay_source_target_C_D.png`

## 12. Verdict
Stage 5A-128 Verdict: **GO**

*Note: This is still a small-set overfit pilot, not full-training success.*

## 13. Next Steps
- We should **not** directly claim full training success yet.
- The next step should be to discuss whether to enter a more rigorous setting closer to full chair training (e.g., full dataset training) or to first perform trigger/loss diagnostics on these 6 `trigger attack weak` failed sources. Do not automatically run any next stage.
