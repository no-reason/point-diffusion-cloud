# Stage A0: Asset Inventory Report

## 实验资产清单
1. **Clean model checkpoint path**: `/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
2. **Backdoored S2 model checkpoint path**: `/data/personal_data/zyy/point-diffusion-cloud/logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/checkpoints/best_conditional.pt`
3. **S2 target path**: `/data/personal_data/zyy/point-diffusion-cloud/targets/stageC8E_fixed_airplane_target.npy`
4. **S2 metrics_best.json path**: `/data/personal_data/zyy/point-diffusion-cloud/logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/metrics_best.json`
5. **S2 per_source_metrics_best.csv path**: `/data/personal_data/zyy/point-diffusion-cloud/logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/per_source_metrics_best.csv`
6. **S2 visualization path**: `/data/personal_data/zyy/point-diffusion-cloud/results_stageS2_small_sphere_to_airplane/visualizations/`
7. **S2 training source ids / source indices**: 根据 `selected_sources.json` 的解析结果，训练使用了 `001` 到 `128` 对应的 `.npy` 预采样文件。
8. **H5 dataset path**: `/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k_chair_airplane.h5`
9. **trigger implementation file**: `tools/input_triggers.py` 和 `tools/sphere.py`。
10. **当前 git status --short**: 在后台 PVD 修改之后的 working tree (clean diffusion, vae, triggers 等若干 modified files)。
11. **git log -1 --oneline**: 暂无直接影响，主仓库保持本地 working tree 独立运行。

## 关于 Held-out Sources 的选择
**分析**: S2 阶段的 128 个训练样本是特定的独立 `.npy` 文件，并不直接来自于 `H5 Dataset` 的连续索引。
为了确保绝对的未见泛化（Held-out Generalization），我们将按照您的建议：
**WARNING**: 使用 H5 dataset (`chair` 类别，`test` split) 中的 index `128:255` 作为 Provisional Held-out Source 集合，以绝对保证这 128 个样本与 S2 阶段的训练集不会重合。
