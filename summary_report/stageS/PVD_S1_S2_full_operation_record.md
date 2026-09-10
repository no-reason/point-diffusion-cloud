# PVD, Stage S1, and Stage S2 Full Operation Record
## 一、总要求
已阅读。
## 二、Git 状态和代码改动
### Point-Diffusion-Cloud 仓库
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
?? generate_full_operation_record.py
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
#### tools/input_triggers.py Diff
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
#### tools/sphere.py
新建文件，定义了 SphereTrigger，在表面基于 random theta/phi 采样点。
#### stageS1_small_sphere_input_trigger.py & stageS2
新建文件，未被 tracking。S1 和 S2 分别从 Stage5A 和 S1 复制而来，仅修改 target_path，GPU devices 以及 trigger_type='small_sphere'。
### PVD 仓库
```
M modules/functional/backend.py
 M train_generation.py
?? datasets/shapenet_h5_pc.py
?? p0a_sanity.log
?? p0a_sanity.pid
?? p0b_clean.log
?? p0b_clean.pid
?? prepare_target_and_trigger.py
?? run_p0a_sanity.sh
?? train_p0b_clean.sh
```
```
9747265 Merge pull request #2 from alexzhou907/add-license-1
```
新增文件: `prepare_target_and_trigger.py`
## 三、PVD 实验操作记录
1. **仓库路径**: `/data/personal_data/zyy/PVD`
2. **Clone 信息**: 现有代码库。Commit: `9747265a5f141e5546fd4f862bfa66aa59f1bd33`
3. **环境信息**: conda `baddiffusion-img`, PyTorch, 默认 CUDA 11。
4. **数据适配**: PVD 原生支持 ShapeNet h5，通过 `--dataset_type h5 --dataroot /data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k_chair_airplane.h5` 适配，点数默认 2048。
5. **Clean PVD 训练**: 
- 命令: 见 `output.log` 中的 Namespace：`bs=8, category='chair', lr=0.0002, niter=151`。
- 训练了约 150 epochs (151 niter)，最新 Checkpoint: `/data/personal_data/zyy/PVD/output/train_generation/2026-07-07-07-03-04/epoch_149.pth`。
- Loss: 收敛至 ~0.02。Visualizations: `syn/epoch_149_samples_eval_all.png`。
6. **PVD Backdoor**: 
PVD backdoor training has not started yet. (仅生成了目标和 trigger r)
## 四、Stage S1 操作记录
1. **训练脚本路径**: `stageS1_small_sphere_input_trigger.py`
2. **训练命令原文**: `export CUDA_VISIBLE_DEVICES=0; python stageS1_small_sphere_input_trigger.py --checkpoint logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt --target_path targets/stage3_fixed_chair_target.npy --output_dir logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128 --num_sources 128 --trigger_type small_sphere --n_trigger 200 --trigger_scale 0.05`
3. **使用 GPU**: 0
4. **CUDA_VISIBLE_DEVICES**: 0
5. **Clean Checkpoint**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
6. **Target Path**: `targets/stage3_fixed_chair_target.npy`
7. **Output Dir**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128`
8. **日志**: `task-XXXX.log` (stdout)
9. **Checkpoint**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/ckpt_0.945312_5000.pt`
10. **Metrics**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/metrics_best.json`
11. **CSV**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/per_source_metrics_best.csv`
12. **Visualizations**: `results_stageS1_small_sphere_input_trigger/visualizations/`
13. **Final Report**: `summary_report/stageS/stageS1_small_sphere_input_trigger.md`
- 沿用 Stage5A 代码主体，替换了 trigger_type，未触发 fallback 0.10，未误用 checkpoint。
## 五、Stage S2 操作记录
1. **Target Audit Script**: `audit_stageS2_target.py`
2. **Audit Cmd**: `python audit_stageS2_target.py`
3. **Audit Report**: `summary_report/stageS/stageS2_airplane_target_audit.md`
4. **Training Script**: `stageS2_small_sphere_input_trigger.py`
5. **Training Cmd**: `export CUDA_VISIBLE_DEVICES=2; python stageS2_small_sphere_input_trigger.py --checkpoint ... --target_path targets/stageC8E_fixed_airplane_target.npy --output_dir logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128 --trigger_type small_sphere --n_trigger 200 --trigger_scale 0.05`
6. **CUDA_VISIBLE_DEVICES**: 2
7. **Physical GPU**: 2
8-14. **Paths**: `logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/...`
15-18. **Vis**: `plot_visuals_stageS2_airplane.py` -> `results_stageS2_small_sphere_to_airplane/visualizations/` -> report `summary_report/stageS/stageS2_small_sphere_to_airplane.md`
- 严格使用 GPU 2，从 clean checkpoint 初始化（未使用 S1 checkpoint），严格纯 input-space trigger。
## 六、small_sphere trigger 具体实现
1. **路径**: `tools/sphere.py` 和 `tools/input_triggers.py`
2. **类名**: `SphereTrigger`
3. **调用**: `apply_input_trigger(x, trigger_type='small_sphere')`
4. **Center**: `[0.9, -0.9, -0.9]`
5. **Radius**: 0.05
6. **n_trigger**: 200
7. **点生成**: 球面点
8. **随机种子**: `trigger_obj.rng = np.random.default_rng(seed)`
9. **Batch 同一性**: 所有样本被 `expand(B, K, 3)` 覆盖同一个 trigger。
10. **Placement**: `replace_last_K`
11. **Range**: `[-K:]` (后 200 个点)
12. **点数不变**: True, 保持 2048。
13. **Clean 无毒**: 是
14. **Triggered 有毒**: 是
## 七、target 审计记录
### S1 Target: targets/stage3_fixed_chair_target.npy
- shape: (2048, 3)
- dtype: float32
- finite_ratio: 1.0
- min: -1.0
- max: 1.0
- mean: [-0.30648804 -0.30763257  0.00612537]
- std: 0.5571257472038269
- centroid: [-0.30648804 -0.30763257  0.00612537]
- bbox_size: [1.7142687 1.3252145 2.       ]
### S2 Target: targets/stageC8E_fixed_airplane_target.npy
- shape: (2048, 3)
- dtype: float32
- finite_ratio: 1.0
- min: -0.36017999053001404
- max: 0.35673001408576965
- mean: [-0.0298198  -0.0350715  -0.00330972]
- std: 0.10862397402524948
- centroid: [-0.0298198  -0.0350715  -0.00330972]
- bbox_size: [0.65287 0.20869 0.71691]
S2 Target 为 Airplane 已经在审计报告中通过类别特征及来源文件确认无误。
## 八、训练 loss 和配置记录
S1/S2 reuse Stage 5A input-trigger training objective.
- Clean branch: L_clean = || VAE(x) - ... || (包含 chamfer / kl / codebook)
- Poison branch: L_bd = || VAE(T_sphere(x)) - target ||
- Total loss = L_clean + lambda_bd * L_bd
- Poison rate: implicit in batch ratio (lambda_bd scales it)
- lambda_bd: 2.0 (from 5A)
- batch size: likely 32 / 16 depending on args
- max_iters: 5000
- Optimizer: Adam
## 九、评估逻辑和 ASR 计算记录
C = backdoored model + clean input x
D = backdoored model + triggered input T_sphere(x)
- C 没有 trigger; D 有 trigger
- C/D 使用相同 source index 和相同 target (计算目标CD时)
- ASR 分母为 128 (对 128 个源做全量测试)
```python
success_relaxed = (D_target < C_target and D_target < D_source)
success_margin_005 = (D_target < C_target - 0.05 and D_target < D_source)
```
代码逻辑与此严格一致，详见 `generate_report_stageS2_airplane.py`。
## 十、结果读取记录
### S1
- C_source mean: 0.0148
- C_target mean: 0.2395
- D_source mean: 0.2292
- D_target mean: 0.0224
- ASR: 91.41%
### S2
- C_source mean: 0.0138
- C_target mean: 0.2987
- D_source mean: 0.1248
- D_target mean: 0.0397
- ASR Relaxed: 84.38%
- ASR Margin: 84.38%

**S1 CSV (First 5)**
```csv
source_id,A_source,A_target,B_source,B_target,C_source,C_target,D_source,D_target,finite_ratio_A,finite_ratio_B,finite_ratio_C,finite_ratio_D,clean_preservation_margin,trigger_target_margin,conditional_margin,baseline_gain,success
001,0.7090985178947449,0.7797918319702148,0.7303344011306763,0.8460615277290344,0.0082665681838989,0.2106555700302124,0.2189755588769912,0.0082188080996274,1.0,1.0,1.0,1.0,0.2023890018463134,0.2107567507773637,0.2024367619305849,0.8378427196294069,True
002,0.7779734134674072,0.8459778428077698,0.7942562699317932,0.9213575720787048,0.0067477077245712,0.271441102027893,0.2802889943122864,0.0079795774072408,1.0,1.0,1.0,1.0,0.2646933943033218,0.2723094169050455,0.2634615246206522,0.913377994671464,True
003,0.7767960429191589,0.8922055959701538,0.7121144533157349,0.8648629784584045,0.0188207104802131,0.2788519859313965,0.3146000504493713,0.0080505972728133,1.0,1.0,1.0,1.0,0.2600312754511833,0.306549453176558,0.2708013886585831,0.8568123811855912,True
004,0.6156492233276367,0.8157734870910645,0.6451698541641235,0.8385671377182007,0.0099535658955574,0.2096693813800811,0.1949538886547088,0.0100135039538145,1.0,1.0,1.0,1.0,0.1997158154845237,0.1849403847008943,0.1996558774262666,0.8285536337643862,True
005,0.7786009311676025,0.9200617074966432,0.7241225242614746,0.9072608351707458,0.0132581628859043,0.3067431449890136,0.3412705659866333,0.0081127109006047,1.0,1.0,1.0,1.0,0.2934849821031093,0.3331578550860286,0.2986304340884089,0.8991481242701411,True
```

**S2 CSV (First 5)**
```csv
source_id,A_source,A_target,B_source,B_target,C_source,C_target,D_source,D_target,finite_ratio_A,finite_ratio_B,finite_ratio_C,finite_ratio_D,clean_preservation_margin,trigger_target_margin,conditional_margin,baseline_gain,success_relaxed,success_margin,success
001,0.7090985178947449,2.070002555847168,0.7303344011306763,2.145292043685913,0.0086197666823863,0.2990804314613342,0.143293097615242,0.0037571957800537,1.0,1.0,1.0,1.0,0.2904606647789478,0.1395359018351882,0.2953232356812805,2.14153484790586,True,True,True
002,0.7779734134674072,2.088035345077514,0.7942562699317932,2.1457161903381348,0.0063483221456408,0.2530843317508697,0.1674378216266632,0.0026280516758561,1.0,1.0,1.0,1.0,0.2467360096052289,0.1648097699508071,0.2504562800750136,2.1430881386622787,True,True,True
003,0.7767960429191589,2.1207311153411865,0.7121144533157349,2.1122279167175293,0.0188223589211702,0.2551897764205932,0.1365767866373062,0.0029300337191671,1.0,1.0,1.0,1.0,0.236367417499423,0.1336467529181391,0.2522597427014261,2.109297882998362,True,True,True
004,0.6156492233276367,2.0693421363830566,0.6451698541641235,2.019716262817383,0.0098562119528651,0.3080061376094818,0.0677654072642326,0.006826058961451,1.0,1.0,1.0,1.0,0.2981499256566167,0.0609393483027815,0.3011800786480307,2.012890203855932,True,True,True
005,0.7786009311676025,2.1328446865081787,0.7241225242614746,2.0814876556396484,0.0122584141790866,0.2355057001113891,0.1048798263072967,0.0040300218388438,1.0,1.0,1.0,1.0,0.2232472859323024,0.1008498044684529,0.2314756782725453,2.0774576338008046,True,True,True
```
## 十一、可视化记录
- 包含 source / triggered_input / target / C / D
- 图像数量: 各生成了 grid, top_success_cases_C_D, 和 sample 单独图像
S1 观察: C 依旧为 chair-like, D 为 fixed-chair, trigger 在 triggered_input 边缘可见。
S2 观察: C 依旧为 chair-like, D 呈现清晰 airplane-like（非随机崩坏），trigger 在边缘以 0.05 极小圆球出现，极具隐蔽性。
## 十二、污染检查
全局搜索 `target_r`, `shift_mean`, `epsilon_bd`, `X_T+r` 等关键词，S1/S2 中未参与实际训练。`target_r` 仅用于 assert `None` 以确保无毒。
无互换、无覆盖、无误用 checkpoint 发生。
## 十三、最终操作审计结论
1. PVD clean checkpoint 是否可用: PASS
2. PVD backdoor 是否已开始: NOT_STARTED
3. S1 操作过程是否可信: PASS
4. S2 操作过程是否可信: PASS
5. 所有 warning 列表: None
No evidence of target mismatch, trigger leakage into clean input, diffusion-state trigger contamination, checkpoint misuse, output overwrite, or ASR computation error was found.