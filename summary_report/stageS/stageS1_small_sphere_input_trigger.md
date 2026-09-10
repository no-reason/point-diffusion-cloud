# Stage S1: Small Sphere Input-Trigger Backdoor Pilot

## 1. 实验目的
验证在无 VAE 瓶颈的扩散模型后门链路 (Direction B) 中，将 trigger 几何从 `torus` 替换为 `small_sphere` 能否同样实现稳定的后门攻击。这是纯粹的 trigger geometry ablation。

## 2. Small Sphere Trigger 审计 (Audit)
- **代码实现**: `tools/sphere.py` 中的 `SphereTrigger` 类。
- **配置**:
  - `trigger_type`: small_sphere
  - `n_trigger`: 200 (与 Stage 5A 一致)
  - `sphere_radius / trigger_scale`: 0.05
  - `center`: `[0.9, -0.9, -0.9]` (固定 Universal Center)
- **注入策略 (Placement)**: `replace_last_K`。仅替换点云数组最后 200 个点。
- **采样方式**: 采用球面随机采样 (Surface sampling, `phi/theta`)。为了审计和评测的一致性，在 `apply_input_trigger` 中使用 `np.random.default_rng(seed)` 固定了随机种子。
- **结构不变性**: 输入点数始终保持 `[B, 2048, 3]`，未改变坐标系归一化边界。

## 3. 训练配置 (同 Stage 5A-128)
- **Target**: `targets/stage3_fixed_chair_target.npy` (Fixed Chair)
- **Num Sources**: 128
- **Loss Setup**: `lambda_clean = 10`, `lambda_bd = 2`
- **Poison Rate**: 0.2
- **Training Mode**: `eval_mode_training_inherited_from_stage4b1`
- **唯一变化**: `trigger_type` 从 `torus` 改为 `small_sphere`。

## 4. 核心对比结果表格

| 触发器类型 (128 Sources) | C_source mean (越低越好) | C_target mean (越高越好) | D_source mean (越高越好) | D_target mean (越低越好) | ASR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Torus (Stage 5A)** | 0.0136 | 0.2453 | 0.2294 | 0.0146 | 95.31% |
| **Small Sphere (Stage S1)** | 0.0148 | 0.2395 | 0.2292 | 0.0224 | 91.41% |

**指标分析**:
*(请观察上方表格，确认 Sphere 是否与 Torus 表现相当)*

## 5. 详细逐样本成功情况 (Per-source Success Table)

| source_id | C_source | C_target | D_source | D_target | B_target | success | fail_reason |
|-----------|----------|----------|----------|----------|----------|---------|-------------|
| 001 | 0.0083 | 0.2107 | 0.2190 | 0.0082 | 0.8461 | True | N/A |
| 002 | 0.0067 | 0.2714 | 0.2803 | 0.0080 | 0.9214 | True | N/A |
| 003 | 0.0188 | 0.2789 | 0.3146 | 0.0081 | 0.8649 | True | N/A |
| 004 | 0.0100 | 0.2097 | 0.1950 | 0.0100 | 0.8386 | True | N/A |
| 005 | 0.0133 | 0.3067 | 0.3413 | 0.0081 | 0.9073 | True | N/A |
| 006 | 0.0066 | 0.2984 | 0.3131 | 0.0080 | 0.9143 | True | N/A |
| 007 | 0.0204 | 0.1575 | 0.0213 | 0.1518 | 0.6490 | False | trigger attack weak / insufficient target attraction |
| 008 | 0.0241 | 0.1064 | 0.0585 | 0.0557 | 0.7822 | True | N/A |
| 009 | 0.0498 | 0.1107 | 0.1176 | 0.0125 | 0.8578 | True | N/A |
| 010 | 0.0077 | 0.2798 | 0.2614 | 0.0083 | 0.9024 | True | N/A |
| 011 | 0.0094 | 0.2245 | 0.2143 | 0.0079 | 0.8683 | True | N/A |
| 012 | 0.0180 | 0.2846 | 0.1261 | 0.0345 | 0.9447 | True | N/A |
| 013 | 0.0259 | 0.1612 | 0.0284 | 0.1573 | 0.7843 | False | trigger attack weak / insufficient target attraction |
| 014 | 0.0075 | 0.2561 | 0.3022 | 0.0083 | 0.9212 | True | N/A |
| 015 | 0.0091 | 0.2009 | 0.2355 | 0.0082 | 0.7980 | True | N/A |
| dataset_16 | 0.0070 | 0.2293 | 0.2396 | 0.0081 | 0.9038 | True | N/A |
| dataset_17 | 0.0092 | 0.1991 | 0.1832 | 0.0084 | 0.9060 | True | N/A |
| dataset_18 | 0.0255 | 0.1766 | 0.0861 | 0.0330 | 0.8854 | True | N/A |
| dataset_19 | 0.0093 | 0.2860 | 0.1775 | 0.0122 | 0.9496 | True | N/A |
| dataset_20 | 0.0081 | 0.3718 | 0.3570 | 0.0075 | 0.8807 | True | N/A |
| dataset_21 | 0.0236 | 0.2103 | 0.2479 | 0.0073 | 0.8261 | True | N/A |
| dataset_22 | 0.0072 | 0.2825 | 0.2767 | 0.0082 | 0.9006 | True | N/A |
| dataset_23 | 0.0078 | 0.3345 | 0.3237 | 0.0076 | 0.8339 | True | N/A |
| dataset_24 | 0.0435 | 0.1884 | 0.1726 | 0.0131 | 0.8517 | True | N/A |
| dataset_25 | 0.0150 | 0.4050 | 0.4569 | 0.0085 | 0.9735 | True | N/A |
| dataset_26 | 0.0178 | 0.1320 | 0.0497 | 0.0930 | 0.8625 | False | trigger attack weak / insufficient target attraction |
| dataset_27 | 0.0292 | 0.2493 | 0.2400 | 0.0168 | 0.9881 | True | N/A |
| dataset_28 | 0.0072 | 0.3172 | 0.3587 | 0.0076 | 0.8911 | True | N/A |
| dataset_29 | 0.0365 | 0.2208 | 0.0358 | 0.2246 | 0.6744 | False | trigger attack weak / insufficient target attraction |
| dataset_30 | 0.0183 | 0.2830 | 0.1425 | 0.0433 | 0.9299 | True | N/A |
| dataset_31 | 0.0116 | 0.2423 | 0.2944 | 0.0090 | 0.8655 | True | N/A |
| dataset_32 | 0.0181 | 0.2466 | 0.3431 | 0.0100 | 0.9431 | True | N/A |
| dataset_33 | 0.0110 | 0.1541 | 0.1360 | 0.0093 | 0.8368 | True | N/A |
| dataset_34 | 0.0330 | 0.1986 | 0.0853 | 0.0950 | 1.0189 | False | trigger attack weak / insufficient target attraction |
| dataset_35 | 0.0168 | 0.1893 | 0.1542 | 0.0153 | 0.8651 | True | N/A |
| dataset_36 | 0.0138 | 0.2720 | 0.1908 | 0.0107 | 0.9301 | True | N/A |
| dataset_37 | 0.0139 | 0.1730 | 0.1656 | 0.0187 | 0.9145 | True | N/A |
| dataset_38 | 0.0118 | 0.2459 | 0.1672 | 0.0137 | 0.9104 | True | N/A |
| dataset_39 | 0.0775 | 0.0826 | 0.0840 | 0.0751 | 1.0186 | True | N/A |
| dataset_40 | 0.0087 | 0.2822 | 0.3099 | 0.0081 | 0.8889 | True | N/A |
| dataset_41 | 0.0309 | 0.2303 | 0.1268 | 0.0634 | 0.9638 | True | N/A |
| dataset_42 | 0.0105 | 0.2748 | 0.1679 | 0.0161 | 0.9516 | True | N/A |
| dataset_43 | 0.0145 | 0.2562 | 0.2356 | 0.0105 | 0.9513 | True | N/A |
| dataset_44 | 0.0207 | 0.1071 | 0.0902 | 0.0112 | 0.8663 | True | N/A |
| dataset_45 | 0.0098 | 0.1614 | 0.1663 | 0.0084 | 0.8312 | True | N/A |
| dataset_46 | 0.0076 | 0.3630 | 0.3801 | 0.0078 | 0.9632 | True | N/A |
| dataset_47 | 0.0116 | 0.1830 | 0.2112 | 0.0080 | 0.8225 | True | N/A |
| dataset_48 | 0.0078 | 0.2063 | 0.2384 | 0.0089 | 0.8824 | True | N/A |
| dataset_49 | 0.0100 | 0.2907 | 0.3224 | 0.0076 | 0.8816 | True | N/A |
| dataset_50 | 0.0060 | 0.3696 | 0.3868 | 0.0080 | 0.9274 | True | N/A |
| dataset_51 | 0.0089 | 0.3276 | 0.2417 | 0.0110 | 0.9670 | True | N/A |
| dataset_52 | 0.0152 | 0.4463 | 0.5473 | 0.0090 | 0.9206 | True | N/A |
| dataset_53 | 0.0324 | 0.1501 | 0.1341 | 0.0199 | 0.9660 | True | N/A |
| dataset_54 | 0.0111 | 0.2763 | 0.3172 | 0.0079 | 0.8658 | True | N/A |
| dataset_55 | 0.0120 | 0.5266 | 0.4923 | 0.0113 | 0.8289 | True | N/A |
| dataset_56 | 0.0069 | 0.3509 | 0.3462 | 0.0076 | 0.9029 | True | N/A |
| dataset_57 | 0.0078 | 0.2490 | 0.2842 | 0.0080 | 0.9312 | True | N/A |
| dataset_58 | 0.0084 | 0.2645 | 0.2736 | 0.0077 | 0.8522 | True | N/A |
| dataset_59 | 0.0195 | 0.3244 | 0.1383 | 0.0594 | 0.9502 | True | N/A |
| dataset_60 | 0.0114 | 0.4992 | 0.5207 | 0.0086 | 0.9324 | True | N/A |
| dataset_61 | 0.0126 | 0.1826 | 0.2246 | 0.0086 | 0.8200 | True | N/A |
| dataset_62 | 0.0110 | 0.2890 | 0.2960 | 0.0090 | 0.9258 | True | N/A |
| dataset_63 | 0.0064 | 0.2450 | 0.2710 | 0.0080 | 0.8880 | True | N/A |
| dataset_64 | 0.0072 | 0.3458 | 0.3988 | 0.0078 | 0.8533 | True | N/A |
| dataset_65 | 0.0088 | 0.1534 | 0.1717 | 0.0082 | 0.8087 | True | N/A |
| dataset_66 | 0.0079 | 0.2477 | 0.2473 | 0.0089 | 0.9273 | True | N/A |
| dataset_67 | 0.0189 | 0.0819 | 0.0446 | 0.0624 | 0.8127 | False | trigger attack weak / insufficient target attraction |
| dataset_68 | 0.0089 | 0.2446 | 0.2396 | 0.0092 | 0.9332 | True | N/A |
| dataset_69 | 0.0116 | 0.1907 | 0.2272 | 0.0083 | 0.9032 | True | N/A |
| dataset_70 | 0.0082 | 0.2515 | 0.2772 | 0.0089 | 0.8739 | True | N/A |
| dataset_71 | 0.0147 | 0.0723 | 0.0609 | 0.0128 | 0.7520 | True | N/A |
| dataset_72 | 0.0184 | 0.1091 | 0.0577 | 0.0165 | 0.8731 | True | N/A |
| dataset_73 | 0.0214 | 0.0656 | 0.0321 | 0.0283 | 0.6913 | True | N/A |
| dataset_74 | 0.0067 | 0.3551 | 0.3634 | 0.0078 | 0.9584 | True | N/A |
| dataset_75 | 0.0072 | 0.3446 | 0.3635 | 0.0077 | 0.8666 | True | N/A |
| dataset_76 | 0.0060 | 0.3188 | 0.3464 | 0.0079 | 0.9105 | True | N/A |
| dataset_77 | 0.0109 | 0.1974 | 0.2036 | 0.0088 | 0.7663 | True | N/A |
| dataset_78 | 0.0116 | 0.3878 | 0.4375 | 0.0084 | 0.9544 | True | N/A |
| dataset_79 | 0.0165 | 0.1077 | 0.0895 | 0.0461 | 0.7588 | True | N/A |
| dataset_80 | 0.0091 | 0.2097 | 0.2105 | 0.0079 | 0.8563 | True | N/A |
| dataset_81 | 0.0080 | 0.3103 | 0.3408 | 0.0086 | 0.9195 | True | N/A |
| dataset_82 | 0.0059 | 0.2579 | 0.3024 | 0.0079 | 0.9165 | True | N/A |
| dataset_83 | 0.0266 | 0.1381 | 0.0464 | 0.1393 | 0.7097 | False | trigger attack weak / insufficient target attraction |
| dataset_84 | 0.0266 | 0.2981 | 0.2095 | 0.0210 | 0.8722 | True | N/A |
| dataset_85 | 0.0105 | 0.2354 | 0.2154 | 0.0077 | 0.9060 | True | N/A |
| dataset_86 | 0.0576 | 0.1055 | 0.0692 | 0.0975 | 0.9400 | False | trigger attack weak / insufficient target attraction |
| dataset_87 | 0.0245 | 0.1679 | 0.0251 | 0.1605 | 0.6065 | False | trigger attack weak / insufficient target attraction |
| dataset_88 | 0.0096 | 0.3251 | 0.2755 | 0.0078 | 0.8611 | True | N/A |
| dataset_89 | 0.0170 | 0.3101 | 0.2464 | 0.0182 | 1.0062 | True | N/A |
| dataset_90 | 0.0098 | 0.2556 | 0.1855 | 0.0109 | 0.9824 | True | N/A |
| dataset_91 | 0.0101 | 0.1964 | 0.2051 | 0.0081 | 0.8179 | True | N/A |
| dataset_92 | 0.0085 | 0.2262 | 0.3173 | 0.0080 | 0.8713 | True | N/A |
| dataset_93 | 0.0071 | 0.3009 | 0.3052 | 0.0077 | 0.9243 | True | N/A |
| dataset_94 | 0.0083 | 0.2332 | 0.1998 | 0.0087 | 0.8672 | True | N/A |
| dataset_95 | 0.0116 | 0.1923 | 0.2021 | 0.0093 | 0.8177 | True | N/A |
| dataset_96 | 0.0106 | 0.4174 | 0.4041 | 0.0090 | 0.9572 | True | N/A |
| dataset_97 | 0.0128 | 0.1727 | 0.2022 | 0.0081 | 0.7808 | True | N/A |
| dataset_98 | 0.0172 | 0.0847 | 0.0419 | 0.0806 | 0.6843 | False | trigger attack weak / insufficient target attraction |
| dataset_99 | 0.0249 | 0.0824 | 0.0702 | 0.0275 | 0.8124 | True | N/A |
| dataset_100 | 0.0350 | 0.0611 | 0.0520 | 0.0601 | 0.6556 | False | trigger attack weak / insufficient target attraction |
| dataset_101 | 0.0135 | 0.2807 | 0.1740 | 0.0154 | 0.9875 | True | N/A |
| dataset_102 | 0.0196 | 0.0611 | 0.0886 | 0.0100 | 0.8840 | True | N/A |
| dataset_103 | 0.0073 | 0.2989 | 0.3505 | 0.0079 | 0.9392 | True | N/A |
| dataset_104 | 0.0073 | 0.2812 | 0.3642 | 0.0088 | 0.9306 | True | N/A |
| dataset_105 | 0.0120 | 0.1756 | 0.1392 | 0.0099 | 0.8237 | True | N/A |
| dataset_106 | 0.0200 | 0.1656 | 0.2484 | 0.0099 | 0.9869 | True | N/A |
| dataset_107 | 0.0116 | 0.3088 | 0.3412 | 0.0079 | 0.9033 | True | N/A |
| dataset_108 | 0.0070 | 0.3082 | 0.3558 | 0.0084 | 0.9564 | True | N/A |
| dataset_109 | 0.0221 | 0.2321 | 0.2116 | 0.0116 | 0.9698 | True | N/A |
| dataset_110 | 0.0255 | 0.4238 | 0.5014 | 0.0092 | 0.9879 | True | N/A |
| dataset_111 | 0.0142 | 0.1363 | 0.1303 | 0.0104 | 0.8076 | True | N/A |
| dataset_112 | 0.0111 | 0.2661 | 0.2765 | 0.0079 | 0.8838 | True | N/A |
| dataset_113 | 0.0088 | 0.2304 | 0.3076 | 0.0083 | 0.8609 | True | N/A |
| dataset_114 | 0.0098 | 0.1368 | 0.1479 | 0.0099 | 0.7828 | True | N/A |
| dataset_115 | 0.0095 | 0.1686 | 0.1745 | 0.0081 | 0.7984 | True | N/A |
| dataset_116 | 0.0117 | 0.2449 | 0.1881 | 0.0113 | 0.9353 | True | N/A |
| dataset_117 | 0.0064 | 0.3917 | 0.4185 | 0.0080 | 0.9656 | True | N/A |
| dataset_118 | 0.0186 | 0.0530 | 0.0615 | 0.0180 | 0.7414 | True | N/A |
| dataset_119 | 0.0088 | 0.2437 | 0.1513 | 0.0109 | 0.8992 | True | N/A |
| dataset_120 | 0.0065 | 0.2833 | 0.3407 | 0.0080 | 0.9211 | True | N/A |
| dataset_121 | 0.0080 | 0.3911 | 0.4213 | 0.0078 | 0.8611 | True | N/A |
| dataset_122 | 0.0160 | 0.1333 | 0.0574 | 0.0150 | 0.8739 | True | N/A |
| dataset_123 | 0.0188 | 0.1279 | 0.0753 | 0.0159 | 0.9632 | True | N/A |
| dataset_124 | 0.0079 | 0.2927 | 0.3416 | 0.0080 | 0.9757 | True | N/A |
| dataset_125 | 0.0067 | 0.3223 | 0.3561 | 0.0079 | 0.9585 | True | N/A |
| dataset_126 | 0.0203 | 0.1550 | 0.1087 | 0.0323 | 0.8975 | True | N/A |
| dataset_127 | 0.0073 | 0.3632 | 0.3832 | 0.0078 | 1.0056 | True | N/A |
| dataset_128 | 0.0109 | 0.2240 | 0.2159 | 0.0093 | 0.9550 | True | N/A |

## 6. 可视化路径 (Visualizations)
- 原始点云与 Trigger 对比: `results_stageS1_small_sphere_input_trigger/visualizations/source_trigger_target_grid.png`
- 成功组生成结果: `results_stageS1_small_sphere_input_trigger/visualizations/top_success_cases_C_D.png`
- 失败组生成结果 (若有): `results_stageS1_small_sphere_input_trigger/visualizations/failed_cases_C_D_part1.png`
*(请在报告下方查阅人工验证的隐蔽性对比)*

## 7. 最终判决 (Final Verdict)
**GO_SPHERE_TRIGGER**

