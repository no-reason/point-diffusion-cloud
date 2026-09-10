# Stage S2: Small Sphere Input-Trigger To-Airplane Pilot

## 1. 实验目的
在 input-space trigger geometry ablation 中，把 `torus` 替换成 `small_sphere`，并将 target 更改为 `airplane`，测试 Chair -> Airplane 的跨类别输入触发后门。

## 2. GPU 信息
- **Physical GPU Used**: 2
- **CUDA_VISIBLE_DEVICES**: 2
- **Torch Device Count**: 1

## 3. Airplane Target Audit
- **来源**: `targets/stageC8E_fixed_airplane_target.npy`
- **点数**: 2048
- **尺度**: 正常归一化 (-0.3602 到 0.3567)
- 详细见 `summary_report/stageS/stageS2_airplane_target_audit.md`。

## 4. Small Sphere Trigger 审计 (Audit)
- **代码实现**: `tools/sphere.py` 中的 `SphereTrigger` 类。
- **配置**:
  - `trigger_type`: small_sphere
  - `n_trigger`: 200
  - `sphere_radius / trigger_scale`: 0.05
  - `center`: `[0.9, -0.9, -0.9]` (固定 Universal Center)
- **注入策略 (Placement)**: `replace_last_K`。
- **结构不变性**: 输入点数始终保持 `[B, 2048, 3]`。

## 5. 训练配置
- **Target**: `targets/stageC8E_fixed_airplane_target.npy` (Fixed Airplane)
- **Num Sources**: 128
- **Loss Setup**: `lambda_clean = 10`, `lambda_bd = 2`
- **唯一变化**: `trigger_type` 改为 `small_sphere`，Target 改为 `airplane`。

## 6. 核心对比结果表格

| 触发器类型 (128 Sources) | C_source mean (越低越好) | C_target mean (越高越好) | D_source mean (越高越好) | D_target mean (越低越好) | ASR (Relaxed) | ASR (Margin 0.05) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Torus (Stage 5A, Chair->Chair)** | 0.0136 | 0.2453 | 0.2294 | 0.0146 | 95.31% | N/A |
| **Small Sphere (Stage S2, Chair->Air)** | 0.0138 | 0.2987 | 0.1248 | 0.0397 | 84.38% | 84.38% |

**指标分析**:
注意：Stage 5A 是同类别，S2 是跨类别，所以绝对数值不可直接比较，请观察各指标趋势（例如 D_target 是否能被压到极低）。

## 7. 详细逐样本成功情况 (Per-source Success Table)

| source_id | C_source | C_target | D_source | D_target | success_relaxed | success_margin | fail_reason |
|-----------|----------|----------|----------|----------|-----------------|----------------|-------------|
| 001 | 0.0086 | 0.2991 | 0.1433 | 0.0038 | True | True | N/A |
| 002 | 0.0063 | 0.2531 | 0.1674 | 0.0026 | True | True | N/A |
| 003 | 0.0188 | 0.2552 | 0.1366 | 0.0029 | True | True | N/A |
| 004 | 0.0099 | 0.3080 | 0.0678 | 0.0068 | True | True | N/A |
| 005 | 0.0123 | 0.2355 | 0.1049 | 0.0040 | True | True | N/A |
| 006 | 0.0064 | 0.2610 | 0.1766 | 0.0025 | True | True | N/A |
| 007 | 0.0218 | 0.4330 | 0.0218 | 0.4206 | False | False | trigger attack weak / insufficient target attraction |
| 008 | 0.0180 | 0.4716 | 0.0450 | 0.1760 | False | False | trigger attack weak / insufficient target attraction |
| 009 | 0.0611 | 0.3919 | 0.1814 | 0.0092 | True | True | N/A |
| 010 | 0.0075 | 0.2817 | 0.1373 | 0.0028 | True | True | N/A |
| 011 | 0.0096 | 0.2761 | 0.2297 | 0.0027 | True | True | N/A |
| 012 | 0.0196 | 0.3218 | 0.0404 | 0.0518 | False | False | trigger attack weak / insufficient target attraction |
| 013 | 0.0164 | 0.3432 | 0.0157 | 0.2976 | False | False | trigger attack weak / insufficient target attraction |
| 014 | 0.0075 | 0.2692 | 0.1637 | 0.0023 | True | True | N/A |
| 015 | 0.0079 | 0.3142 | 0.1544 | 0.0028 | True | True | N/A |
| dataset_0 | 0.0160 | 0.6724 | 0.2096 | 0.3866 | False | False | trigger attack weak / insufficient target attraction |
| dataset_16 | 0.0069 | 0.2806 | 0.1768 | 0.0025 | True | True | N/A |
| dataset_17 | 0.0094 | 0.2698 | 0.1885 | 0.0027 | True | True | N/A |
| dataset_18 | 0.0294 | 0.3167 | 0.0424 | 0.0390 | True | True | N/A |
| dataset_19 | 0.0086 | 0.3133 | 0.1015 | 0.0070 | True | True | N/A |
| dataset_20 | 0.0072 | 0.2490 | 0.1806 | 0.0030 | True | True | N/A |
| dataset_21 | 0.0265 | 0.3085 | 0.0935 | 0.0048 | True | True | N/A |
| dataset_22 | 0.0067 | 0.2534 | 0.1795 | 0.0025 | True | True | N/A |
| dataset_23 | 0.0073 | 0.2639 | 0.1387 | 0.0028 | True | True | N/A |
| dataset_24 | 0.0423 | 0.3079 | 0.0486 | 0.0306 | True | True | N/A |
| dataset_25 | 0.0132 | 0.2537 | 0.2582 | 0.0031 | True | True | N/A |
| dataset_26 | 0.0158 | 0.2848 | 0.0293 | 0.0885 | False | False | trigger attack weak / insufficient target attraction |
| dataset_27 | 0.0208 | 0.2695 | 0.0372 | 0.0163 | True | True | N/A |
| dataset_28 | 0.0068 | 0.2681 | 0.1279 | 0.0027 | True | True | N/A |
| dataset_29 | 0.0272 | 0.4532 | 0.0257 | 0.4207 | False | False | trigger attack weak / insufficient target attraction |
| dataset_30 | 0.0253 | 0.3521 | 0.0450 | 0.0447 | True | True | N/A |
| dataset_31 | 0.0140 | 0.2255 | 0.1188 | 0.0033 | True | True | N/A |
| dataset_32 | 0.0232 | 0.2403 | 0.2623 | 0.0028 | True | True | N/A |
| dataset_33 | 0.0114 | 0.3202 | 0.1142 | 0.0068 | True | True | N/A |
| dataset_34 | 0.0237 | 0.2740 | 0.0161 | 0.1290 | False | False | trigger attack weak / insufficient target attraction |
| dataset_35 | 0.0286 | 0.3333 | 0.0516 | 0.0117 | True | True | N/A |
| dataset_36 | 0.0138 | 0.2858 | 0.1006 | 0.0040 | True | True | N/A |
| dataset_37 | 0.0194 | 0.2806 | 0.1373 | 0.0060 | True | True | N/A |
| dataset_38 | 0.0135 | 0.3309 | 0.0496 | 0.0114 | True | True | N/A |
| dataset_39 | 0.0165 | 0.2283 | 0.0135 | 0.1769 | False | False | trigger attack weak / insufficient target attraction |
| dataset_40 | 0.0091 | 0.2586 | 0.1535 | 0.0023 | True | True | N/A |
| dataset_41 | 0.0140 | 0.2431 | 0.0137 | 0.0933 | False | False | trigger attack weak / insufficient target attraction |
| dataset_42 | 0.0115 | 0.3209 | 0.0559 | 0.0105 | True | True | N/A |
| dataset_43 | 0.0157 | 0.2878 | 0.1274 | 0.0039 | True | True | N/A |
| dataset_44 | 0.0192 | 0.3778 | 0.0675 | 0.0193 | True | True | N/A |
| dataset_45 | 0.0095 | 0.3015 | 0.1986 | 0.0033 | True | True | N/A |
| dataset_46 | 0.0071 | 0.2270 | 0.1965 | 0.0029 | True | True | N/A |
| dataset_47 | 0.0112 | 0.3088 | 0.1058 | 0.0084 | True | True | N/A |
| dataset_48 | 0.0064 | 0.3265 | 0.2437 | 0.0024 | True | True | N/A |
| dataset_49 | 0.0103 | 0.2976 | 0.1416 | 0.0025 | True | True | N/A |
| dataset_50 | 0.0050 | 0.2363 | 0.1487 | 0.0033 | True | True | N/A |
| dataset_51 | 0.0102 | 0.3103 | 0.0666 | 0.0065 | True | True | N/A |
| dataset_52 | 0.0180 | 0.2449 | 0.1562 | 0.0034 | True | True | N/A |
| dataset_53 | 0.0190 | 0.2139 | 0.0281 | 0.0399 | False | False | trigger attack weak / insufficient target attraction |
| dataset_54 | 0.0099 | 0.2272 | 0.2452 | 0.0028 | True | True | N/A |
| dataset_55 | 0.0231 | 0.2238 | 0.1590 | 0.0044 | True | True | N/A |
| dataset_56 | 0.0062 | 0.2421 | 0.1561 | 0.0029 | True | True | N/A |
| dataset_57 | 0.0073 | 0.2510 | 0.1628 | 0.0025 | True | True | N/A |
| dataset_58 | 0.0076 | 0.2814 | 0.2106 | 0.0024 | True | True | N/A |
| dataset_59 | 0.0192 | 0.3602 | 0.0177 | 0.0758 | False | False | trigger attack weak / insufficient target attraction |
| dataset_60 | 0.0120 | 0.2495 | 0.2657 | 0.0030 | True | True | N/A |
| dataset_61 | 0.0141 | 0.3143 | 0.2297 | 0.0020 | True | True | N/A |
| dataset_62 | 0.0117 | 0.2686 | 0.1679 | 0.0028 | True | True | N/A |
| dataset_63 | 0.0061 | 0.2881 | 0.1927 | 0.0027 | True | True | N/A |
| dataset_64 | 0.0070 | 0.2560 | 0.1345 | 0.0031 | True | True | N/A |
| dataset_65 | 0.0097 | 0.3062 | 0.1883 | 0.0027 | True | True | N/A |
| dataset_66 | 0.0078 | 0.2282 | 0.2007 | 0.0026 | True | True | N/A |
| dataset_67 | 0.0199 | 0.3905 | 0.0774 | 0.1037 | False | False | trigger attack weak / insufficient target attraction |
| dataset_68 | 0.0099 | 0.2746 | 0.1004 | 0.0025 | True | True | N/A |
| dataset_69 | 0.0104 | 0.2565 | 0.1590 | 0.0039 | True | True | N/A |
| dataset_70 | 0.0087 | 0.2642 | 0.1464 | 0.0023 | True | True | N/A |
| dataset_71 | 0.0129 | 0.4520 | 0.1465 | 0.0139 | True | True | N/A |
| dataset_72 | 0.0215 | 0.2974 | 0.0797 | 0.0331 | True | True | N/A |
| dataset_73 | 0.0230 | 0.5125 | 0.0414 | 0.2986 | False | False | trigger attack weak / insufficient target attraction |
| dataset_74 | 0.0062 | 0.2321 | 0.1776 | 0.0028 | True | True | N/A |
| dataset_75 | 0.0073 | 0.2609 | 0.1180 | 0.0029 | True | True | N/A |
| dataset_76 | 0.0054 | 0.2500 | 0.1784 | 0.0029 | True | True | N/A |
| dataset_77 | 0.0095 | 0.3338 | 0.1530 | 0.0035 | True | True | N/A |
| dataset_78 | 0.0135 | 0.2324 | 0.1128 | 0.0024 | True | True | N/A |
| dataset_79 | 0.0118 | 0.2415 | 0.0646 | 0.0576 | True | True | N/A |
| dataset_80 | 0.0100 | 0.2878 | 0.1346 | 0.0022 | True | True | N/A |
| dataset_81 | 0.0098 | 0.2401 | 0.1940 | 0.0027 | True | True | N/A |
| dataset_82 | 0.0065 | 0.2516 | 0.1465 | 0.0023 | True | True | N/A |
| dataset_83 | 0.0175 | 0.3394 | 0.0242 | 0.2158 | False | False | trigger attack weak / insufficient target attraction |
| dataset_84 | 0.0266 | 0.3180 | 0.0637 | 0.0297 | True | True | N/A |
| dataset_85 | 0.0102 | 0.2946 | 0.1374 | 0.0038 | True | True | N/A |
| dataset_86 | 0.0159 | 0.2202 | 0.0161 | 0.1901 | False | False | trigger attack weak / insufficient target attraction |
| dataset_87 | 0.0214 | 0.5113 | 0.0213 | 0.5021 | False | False | trigger attack weak / insufficient target attraction |
| dataset_88 | 0.0092 | 0.2898 | 0.1101 | 0.0038 | True | True | N/A |
| dataset_89 | 0.0213 | 0.2433 | 0.1162 | 0.0046 | True | True | N/A |
| dataset_90 | 0.0085 | 0.2827 | 0.0922 | 0.0059 | True | True | N/A |
| dataset_91 | 0.0094 | 0.3083 | 0.1080 | 0.0045 | True | True | N/A |
| dataset_92 | 0.0096 | 0.2775 | 0.1375 | 0.0022 | True | True | N/A |
| dataset_93 | 0.0071 | 0.2307 | 0.1985 | 0.0024 | True | True | N/A |
| dataset_94 | 0.0081 | 0.2874 | 0.1466 | 0.0032 | True | True | N/A |
| dataset_95 | 0.0105 | 0.3531 | 0.2184 | 0.0033 | True | True | N/A |
| dataset_96 | 0.0101 | 0.2857 | 0.1854 | 0.0036 | True | True | N/A |
| dataset_97 | 0.0118 | 0.3152 | 0.1498 | 0.0031 | True | True | N/A |
| dataset_98 | 0.0175 | 0.4472 | 0.0386 | 0.3561 | False | False | trigger attack weak / insufficient target attraction |
| dataset_99 | 0.0267 | 0.3495 | 0.1087 | 0.0631 | True | True | N/A |
| dataset_100 | 0.0332 | 0.4563 | 0.0940 | 0.1089 | False | False | trigger attack weak / insufficient target attraction |
| dataset_101 | 0.0126 | 0.2762 | 0.0936 | 0.0054 | True | True | N/A |
| dataset_102 | 0.0167 | 0.3075 | 0.0555 | 0.0397 | True | True | N/A |
| dataset_103 | 0.0068 | 0.2339 | 0.1189 | 0.0025 | True | True | N/A |
| dataset_104 | 0.0083 | 0.2203 | 0.1510 | 0.0024 | True | True | N/A |
| dataset_105 | 0.0150 | 0.3322 | 0.0305 | 0.0254 | True | True | N/A |
| dataset_106 | 0.0179 | 0.2397 | 0.1025 | 0.0037 | True | True | N/A |
| dataset_107 | 0.0142 | 0.2748 | 0.1215 | 0.0023 | True | True | N/A |
| dataset_108 | 0.0067 | 0.2415 | 0.1298 | 0.0023 | True | True | N/A |
| dataset_109 | 0.0186 | 0.2852 | 0.0855 | 0.0051 | True | True | N/A |
| dataset_110 | 0.0232 | 0.2996 | 0.2543 | 0.0082 | True | True | N/A |
| dataset_111 | 0.0162 | 0.3622 | 0.0623 | 0.0139 | True | True | N/A |
| dataset_112 | 0.0100 | 0.2533 | 0.1191 | 0.0029 | True | True | N/A |
| dataset_113 | 0.0099 | 0.2546 | 0.1880 | 0.0028 | True | True | N/A |
| dataset_114 | 0.0088 | 0.3981 | 0.2556 | 0.0034 | True | True | N/A |
| dataset_115 | 0.0096 | 0.3211 | 0.2496 | 0.0029 | True | True | N/A |
| dataset_116 | 0.0140 | 0.2847 | 0.0510 | 0.0047 | True | True | N/A |
| dataset_117 | 0.0053 | 0.2315 | 0.1385 | 0.0037 | True | True | N/A |
| dataset_118 | 0.0207 | 0.4863 | 0.2461 | 0.0297 | True | True | N/A |
| dataset_119 | 0.0091 | 0.3025 | 0.0611 | 0.0170 | True | True | N/A |
| dataset_120 | 0.0067 | 0.2385 | 0.1591 | 0.0025 | True | True | N/A |
| dataset_121 | 0.0070 | 0.2653 | 0.1428 | 0.0036 | True | True | N/A |
| dataset_122 | 0.0169 | 0.3652 | 0.0517 | 0.0292 | True | True | N/A |
| dataset_123 | 0.0203 | 0.3458 | 0.0592 | 0.0216 | True | True | N/A |
| dataset_124 | 0.0076 | 0.2224 | 0.1452 | 0.0022 | True | True | N/A |
| dataset_125 | 0.0059 | 0.2269 | 0.1640 | 0.0025 | True | True | N/A |
| dataset_126 | 0.0216 | 0.3469 | 0.0304 | 0.0823 | False | False | trigger attack weak / insufficient target attraction |
| dataset_127 | 0.0064 | 0.2283 | 0.1449 | 0.0030 | True | True | N/A |

## 8. 可视化路径与人工观察 (Visualizations)
- 原始点云与 Trigger 对比: `results_stageS2_small_sphere_to_airplane/visualizations/source_trigger_target_grid.png`
- 成功组生成结果: `results_stageS2_small_sphere_to_airplane/visualizations/top_success_cases_C_D.png`
- 失败组生成结果 (若有): `results_stageS2_small_sphere_to_airplane/visualizations/failed_cases_C_D_part1.png`

人工核验：
- [ ] clean output C 仍然 chair-like
- [ ] triggered output D 接近 airplane target 且无明显几何崩坏
- [ ] small sphere 比 torus 更隐蔽

## 9. 最终判决 (Final Verdict)
**GO_SPHERE_AIRPLANE**

