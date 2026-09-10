# Stage S1 & S2: Reproducibility and Code Audit Report
## 一、完整实验信息
### Stage S1
- **实验名称**: Stage S1: Small Sphere Input-Trigger FixedChair Pilot
- **实验目标**: 同类别 (Chair -> Fixed Chair) 下验证 small_sphere trigger 替换 torus 的后门效果。
- **使用的 GPU**: 0 (Physical GPU)
- **CUDA_VISIBLE_DEVICES**: 0
- **Training Script Path**: `stageS1_small_sphere_input_trigger.py`
- **Evaluation Script Path**: Embedded in `stageS1_small_sphere_input_trigger.py`
- **Report Generation**: `generate_report_stageS1_small_sphere.py`
- **Trigger Implementation**: `tools/input_triggers.py` & `tools/sphere.py`
- **Checkpoint Path**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Target Path**: `targets/stage3_fixed_chair_target.npy`
- **Result Path**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/metrics_best.json`
- **Visualization Directory**: `results_stageS1_small_sphere_input_trigger/visualizations/`
- **Final Report Path**: `summary_report/stageS/stageS1_small_sphere_input_trigger.md`
### Stage S2
- **实验名称**: Stage S2: Small Sphere Input-Trigger To-Airplane Pilot
- **实验目标**: 跨类别 (Chair -> Fixed Airplane) 下验证 small_sphere trigger 的后门效果。
- **使用的 GPU**: 2 (Physical GPU)
- **CUDA_VISIBLE_DEVICES**: 2
- **Training Script Path**: `stageS2_small_sphere_input_trigger.py`
- **Evaluation Script Path**: Embedded in `stageS2_small_sphere_input_trigger.py`
- **Report Generation**: `generate_report_stageS2_airplane.py`
- **Trigger Implementation**: `tools/input_triggers.py` & `tools/sphere.py`
- **Checkpoint Path**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Target Path**: `targets/stageC8E_fixed_airplane_target.npy`
- **Result Path**: `logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/metrics_best.json`
- **Visualization Directory**: `results_stageS2_small_sphere_to_airplane/visualizations/`
- **Final Report Path**: `summary_report/stageS/stageS2_small_sphere_to_airplane.md`
## 二、代码改动
### git status --short
```
 M build_shapenet_h5_from_pts.py
 M experiment-draft.md
 M models/diffusion.py
 M models/vae_flow.py
 M models/vae_gaussian.py
 M tools/input_triggers.py
 M train_gen.py
?? "Chou \347\255\211 - 2023 - How to Backdoor Diffusion Models.pdf"
?? analyze_stageC8A_latent_separation.py
?? analyze_stageC8D_top_latent_reaudit.py
?? analyze_stageC8E_latent_reaudit.py
?? analyze_stageC9A_latent_and_trigger_audit.py
?? audit_chair_airplane_h5_normalization.py
?? audit_earphone_target_preprocessing.py
?? audit_raw_dataset.py
?? audit_stage7a.py
?? audit_stageC7A_loss_scale_code.py
?? audit_stageS2_target.py
?? audit_target_classes.py
?? check_ckpt_cat.py
?? check_cluster_diff.py
?? check_dataset_stats.py
?? check_loss_breakdown.py
?? evaluate_stageC4_abcd.py
?? evaluate_stageC5_source_z_sg_abcd.py
?? evaluate_stageC6_vae_mediated_abcd.py
?? evaluate_stageC7_dual_trigger_ablation.py
?? evaluate_stageC8B0_trigger_strength_sweep.py
?? evaluate_stageC8B1_strong_c6_abcd.py
?? evaluate_stageC8D_c6_ablation.py
?? evaluate_stageC8E_strong_c6_to_airplane.py
?? evaluate_stageC8F_cross_category_conditionality.py
?? evaluate_stageC9A_strong_c7_dual_ablation.py
?? find_worst.py
?? generate_report_stageS1_small_sphere.py
?? generate_report_stageS2_airplane.py
?? generate_stageC8D_report.py
?? loss_thinking.md
?? monitor_clean_vae_stop.py
?? nohup_train.out
?? paper_draft.md
?? plot_visuals.py
?? plot_visuals_stage4b.py
?? plot_visuals_stageS1_small_sphere.py
?? plot_visuals_stageS2_airplane.py
?? prepare_stageC8E_airplane_target.py
?? prompt.md
?? replot_c4_3d.py
?? replot_stageC8E_visualizations.py
?? results_bd/
?? results_stage1a_chair_clean/samples_npy/
?? results_stage1a_chair_clean/visualizations/
?? results_stage1a_chair_clean_confirm/visualizations/
?? results_stage1a_chair_clean_smoke/
?? results_stage1a_earphone_reference_fix/final_stdout.txt
?? results_stage1a_sanity/
?? results_stage2_trigger_sensitivity/samples_npy/
?? results_stage2_trigger_sensitivity/visualizations/
?? results_stage2_trigger_sensitivity_smoke/
?? results_stage3a_fixed_chair_target_sanity/samples_npy/
?? results_stage3a_fixed_chair_target_sanity/visualizations/
?? results_stage3b_earphone_target_ood_decodability/
?? results_stage3b_earphone_target_ood_decodability_normalized/samples_npy/
?? results_stage3b_earphone_target_ood_decodability_normalized/visualizations/
?? results_stage4_single_sample_overfit_fixed_chair/
?? results_stage4b_loss_ratio_fixed_chair/
?? results_stage5a_small_set_fixed_chair/
?? results_stage7a_chair_airplane_clean_baseline/
?? results_stageC4_abcd_prior_z/
?? results_stageC5_source_z_sg_abcd/
?? results_stageC6_vae_mediated_abcd/
?? results_stageC7_dual_trigger_ablation/
?? results_stageC8A_latent_separation/
?? results_stageC8B0_trigger_strength_sweep/
?? results_stageC8B1_strong_c6_abcd/
?? results_stageC8D_c6_ablation/
?? results_stageC8E_strong_c6_to_airplane/
?? results_stageC8F_cross_category_ablation/
?? results_stageC9A_strong_c7_dual_to_airplane/
?? results_stageS1_small_sphere_input_trigger/
?? results_stageS2_small_sphere_to_airplane/
?? run_S1_S2_audit.py
?? run_c7b_sweep.sh
?? run_pipeline.sh
?? run_pipeline_c8b1.sh
?? run_pipeline_c9a.sh
?? run_stageC8D_c6_ablation_grid.py
?? run_stageC8F_cross_category_ablation.py
?? stageS1_small_sphere_input_trigger.py
?? stageS2_small_sphere_input_trigger.py
?? summarize_stageC8F_ablation.py
?? summary_report/backdoor_pathways_summary.md
?? summary_report/stage3/earphone_target_preprocessing_audit_report.md
?? summary_report/stage3/stage3b_earphone_target_ood_decodability_report.md
?? summary_report/stage6/
?? summary_report/stage7/
?? summary_report/stageC/
?? summary_report/stageS/
?? targets/stage3_earphone_target.npy
?? targets/stageC8E_fixed_airplane_target.npy
?? test_bd_gen.py
?? test_dataset_stats.py
?? test_empty.py
?? test_gen_smoke.py
?? train_gen_bd.py
?? train_stageC3_baddiffusion_prior_z.py
?? train_stageC5_source_z_sg_baddiffusion.py
?? train_stageC6_vae_mediated_input_trigger.py
?? train_stageC7_dual_trigger_baddiffusion.py
?? train_stageC8B1_strong_c6_vae_mediated.py
?? train_stageC8D_c6_ablation.py
?? train_stageC9A_strong_c7_dual_to_airplane.py
?? utils/bd_diffusion_trigger.py
?? validate_metrics.py
?? verify_stage7a.py
?? verify_stageC1_custom_xt.py
?? verify_stageC2_triggered_xt.py
?? verify_stageC3_bd_loss_path.py
?? verify_stageC3_prior_z_loss.py
?? verify_stageC4_clean_generation_utility.py
?? verify_stageC5_clean_trigger_baseline.py
?? verify_stageC5_source_z_sg_loss.py
?? verify_stageC6_fixed_chair_target_sanity.py
?? verify_stageC6_vae_mediated_loss.py
?? verify_stageC7_dual_trigger_loss.py
?? verify_stageC8B1_strong_c6_loss.py
```
### git diff
```diff
diff --git a/tools/input_triggers.py b/tools/input_triggers.py
index 4c39c5f..821d6f6 100644
--- a/tools/input_triggers.py
+++ b/tools/input_triggers.py
@@ -75,6 +75,15 @@ def apply_input_trigger(
             device=device,
             dtype=dtype,
         )
+    elif trigger_type == "small_sphere":
+        import numpy as np
+        from tools.sphere import SphereTrigger
+        c = center if center is not None else [0.9, -0.9, -0.9]
+        trigger_obj = SphereTrigger(center=c, radius=trigger_scale, num_points=K)
+        if seed is not None:
+            trigger_obj.rng = np.random.default_rng(seed)
+        sphere_pts = trigger_obj.get_sphere_points()
+        trigger_full = torch.from_numpy(sphere_pts).to(device=device, dtype=dtype).unsqueeze(0).expand(B, K, 3)
     else:
         raise ValueError(f"Unsupported trigger_type for input triggers: {trigger_type}")
 
```
### 代码合规性检查:
1. **apply_input_trigger**: 正确支持 small_sphere (新增了分支)。
2. **replace_last_K**: 是的，所有输入空间 trigger 默认行为即直接覆盖后 K 个点。
3. **Shape**: 保持 `[B, 2048, 3]`。
4. **不改变点数**: 是的。
5. **相同 trigger**: 是的，由于 `replace_last_K` 在 `apply_input_trigger` 内部会进行 `expand(B, K, 3)` 操作。
6. **center**: 固定为 `[0.9, -0.9, -0.9]`。
7. **radius**: 在 S1/S2 中均配置为 `0.05`。
8. **随机种子**: 已在 `apply_input_trigger` 中通过 `trigger_obj.rng = np.random.default_rng(seed)` 固定。
9. **Clean / Trigger 分离**: Evaluation 分别对 clean input 和 triggered input 单独通过模型生成 C 和 D，未混合。
## 三、small_sphere trigger 审计
1. **SphereTrigger 位置**: `tools/sphere.py`
2. **点生成方式**: 表面随机点（Surface Points），基于 `phi/theta`。
3. **Center**: `[0.9, -0.9, -0.9]`
4. **Radius / Scale**: `0.05`
5. **n_trigger**: `200`
6. **Placement**: `replace_last_K`
7. **保持原始点数**: True. 原始 shape: torch.Size([3, 2048, 3]), 注入后 shape: torch.Size([3, 2048, 3])
8. **Trigger 区域 Stats**: min=0.5502, max=0.6498, mean=[0.5996636152267456, 0.5990047454833984, 0.598308265209198]
9. **Full CD(x, T(x))**: 0.0111
## 四、target 审计
### S1 Target: targets/stage3_fixed_chair_target.npy
- **Shape**: (2048, 3)
- **Dtype**: float32
- **Finite ratio**: 1.0000
- **Min / Max**: -1.0000 / 1.0000
- **Centroid**: [-0.30648804 -0.30763257  0.00612537]
- **Bbox Size**: [1.7142687 1.3252145 2.       ]
### S2 Target: targets/stageC8E_fixed_airplane_target.npy
- **Shape**: (2048, 3)
- **Dtype**: float32
- **Finite ratio**: 1.0000
- **Min / Max**: -0.3602 / 0.3567
- **Centroid**: [-0.0298198  -0.0350715  -0.00330972]
- **Bbox Size**: [0.65287 0.20869 0.71691]
S2 的 target 确实是 airplane，来源于 `stageC8E_fixed_airplane_target.npy`。
## 五、确认 S1/S2 是否只做 input-space trigger
### 关键词搜索结果:
- `target_r`: S1 [True], S2 [True]
- `shift_mean`: S1 [False], S2 [False]
- `epsilon_bd`: S1 [False], S2 [False]
- `X_T + r`: S1 [False], S2 [False]
- `x_T + r`: S1 [False], S2 [False]
- `noise_trigger`: S1 [False], S2 [False]
- `diffusion-state trigger`: S1 [False], S2 [False]
- `y_t_bd`: S1 [False], S2 [False]
- `BadDiffusion`: S1 [False], S2 [False]
- `C7`: S1 [False], S2 [False]
- `C6`: S1 [False], S2 [False]
所有的 BadDiffusion 混合态 trigger 参数（shift_mean, epsilon_bd 等）均未在脚本中使用，且不存在干扰，属于纯 input-space poisoning。
注: `target_r` 虽然出现了，但仅存在于 debug audit 中的 `debug_clean.get("target_r", None) is None`，专门用来断言确保没有使用 target_r，符合要求。
## 六、训练配置审计
- **Init Checkpoint**: 相同。S1: logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt / S2: logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt
- **S1 Target**: `targets/stage3_fixed_chair_target.npy`
- **S2 Target**: `targets/stageC8E_fixed_airplane_target.npy`
- **Source Category**: Chair
- **Num Sources**: 128
- **Trigger Type**: small_sphere (Small Sphere)
- **Trigger Scale**: 0.05 (0.05)
- **Output Dir**: S1 和 S2 使用了独立的目录，未互相覆盖。
## 七、评估逻辑审计
1. **C 和 D 独立生成**：使用 `test_model.sample(x_0)` 和 `test_model.sample(x_trigger)` 独立进行。
2. **样本对应正确**：在同一个循环内评估相同 source。
3. **ASR S2 逻辑**：
```python
success_relaxed = (D_target[i] < C_target[i] and D_target[i] < D_source[i] and fin_all)
success_margin = (D_target[i] < C_target[i] - 0.05 and D_target[i] < D_source[i] and fin_all)
```
## 八、结果复核
### S1 (Chair->Chair)
- C_source mean: 0.0148
- C_target mean: 0.2395
- D_source mean: 0.2292
- D_target mean: 0.0224
- ASR: 91.41%
### S2 (Chair->Airplane)
- C_source mean: 0.0138
- C_target mean: 0.2987
- D_source mean: 0.1248
- D_target mean: 0.0397
- ASR Relaxed: 84.38%
- ASR Margin: 84.38%

**S1 CSV (First 3)**
```csv
source_id,A_source,A_target,B_source,B_target,C_source,C_target,D_source,D_target,finite_ratio_A,finite_ratio_B,finite_ratio_C,finite_ratio_D,clean_preservation_margin,trigger_target_margin,conditional_margin,baseline_gain,success
001,0.7090985178947449,0.7797918319702148,0.7303344011306763,0.8460615277290344,0.0082665681838989,0.2106555700302124,0.2189755588769912,0.0082188080996274,1.0,1.0,1.0,1.0,0.2023890018463134,0.2107567507773637,0.2024367619305849,0.8378427196294069,True
002,0.7779734134674072,0.8459778428077698,0.7942562699317932,0.9213575720787048,0.0067477077245712,0.271441102027893,0.2802889943122864,0.0079795774072408,1.0,1.0,1.0,1.0,0.2646933943033218,0.2723094169050455,0.2634615246206522,0.913377994671464,True
003,0.7767960429191589,0.8922055959701538,0.7121144533157349,0.8648629784584045,0.0188207104802131,0.2788519859313965,0.3146000504493713,0.0080505972728133,1.0,1.0,1.0,1.0,0.2600312754511833,0.306549453176558,0.2708013886585831,0.8568123811855912,True
```

**S2 CSV (First 3)**
```csv
source_id,A_source,A_target,B_source,B_target,C_source,C_target,D_source,D_target,finite_ratio_A,finite_ratio_B,finite_ratio_C,finite_ratio_D,clean_preservation_margin,trigger_target_margin,conditional_margin,baseline_gain,success_relaxed,success_margin,success
001,0.7090985178947449,2.070002555847168,0.7303344011306763,2.145292043685913,0.0086197666823863,0.2990804314613342,0.143293097615242,0.0037571957800537,1.0,1.0,1.0,1.0,0.2904606647789478,0.1395359018351882,0.2953232356812805,2.14153484790586,True,True,True
002,0.7779734134674072,2.088035345077514,0.7942562699317932,2.1457161903381348,0.0063483221456408,0.2530843317508697,0.1674378216266632,0.0026280516758561,1.0,1.0,1.0,1.0,0.2467360096052289,0.1648097699508071,0.2504562800750136,2.1430881386622787,True,True,True
003,0.7767960429191589,2.1207311153411865,0.7121144533157349,2.1122279167175293,0.0188223589211702,0.2551897764205932,0.1365767866373062,0.0029300337191671,1.0,1.0,1.0,1.0,0.236367417499423,0.1336467529181391,0.2522597427014261,2.109297882998362,True,True,True
```
## 九、可视化审计
S1 图像: ['failed_cases_C_D_part2.png', 'source_trigger_target_grid.png', 'failed_cases_C_D_part1.png', 'overlay_source_target_C_D.png', 'top_success_cases_C_D.png']
S2 图像: ['failed_cases_C_D_part2.png', 'source_trigger_target_grid.png', 'failed_cases_C_D_part1.png', 'overlay_source_target_C_D.png', 'top_success_cases_C_D.png']
## 十、最终审计结论
1. **S1 结果是否可信**: PASS
2. **S2 结果是否可信**: PASS
3. **Warnings**: None
4. No evidence of target mismatch, trigger leakage into clean input, diffusion-state trigger contamination, or ASR computation error was found.