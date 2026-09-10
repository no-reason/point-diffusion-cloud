# Stage 3B: Earphone Target OOD Decodability Check Report

## 1. Stage 3B 定位
- 本阶段不是后门训练。
- 本阶段不使用 trigger。
- 本阶段旨在评估 `earphone` target 是否对当前 clean chair-only model 是 OOD (Out-Of-Distribution) target。如果 clean decoder 无法生成 earphone-like target，则后续攻击失败不可归因于 trigger learning 失败。

## 2. Checkpoint 信息
- **Checkpoint Path**: `/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Model Class**: `GaussianVAE` (loaded as `gaussian` via model args)
- **Missing / Unexpected Keys**: 0 missing, 0 unexpected
- **Evaluation Settings**: `model.eval()`, `torch.no_grad()`, `use_encoder_mean=True`

## 3. Target 信息
- **Target Path**: `targets/stage3_earphone_target.npy` (Copied from `target_earphone.npy`)
- **Target Shape**: `[1, 2048, 3]`
- **Target Min**: `-5.415`
- **Target Max**: `1.447`
- **Target Mean**: `-0.000`
- **Target Std**: `0.980`
- **Finite Ratio**: `1.0`

**注**：`target_earphone.npy` 的 target normalization 存在明显异常，其 `min` 达到了 `-5.415`，显著超出了通常的 `[-1, 1]` 边界或 `shape_bbox` 的合理范围。

## 4. Decodability 结果
基于 8 次使用 clean decoder `D(E(target_earphone))` 的采样重建，得到以下 Chamfer Distance (CD) 结果：
- **Mean CD to Earphone**: `0.415`
- **Median CD to Earphone**: `0.412`
- **Best CD to Earphone**: `0.399`
- **Worst CD to Earphone**: `0.440`
- **Finite Ratio (Recon)**: `1.0`
- **Mean CD to Fixed Chair**: `0.813`
- **Best CD to Fixed Chair**: `0.754`

**Visualizations Saved To**: 
- `results_stage3b_earphone_target_ood_decodability/visualizations/earphone_target.png`
- `results_stage3b_earphone_target_ood_decodability/visualizations/earphone_recon_best.png`
- `results_stage3b_earphone_target_ood_decodability/visualizations/earphone_recon_median.png`
- `results_stage3b_earphone_target_ood_decodability/visualizations/earphone_recon_worst.png`
- `results_stage3b_earphone_target_ood_decodability/visualizations/earphone_recon_grid.png`
- `results_stage3b_earphone_target_ood_decodability/visualizations/earphone_vs_fixed_chair_reference.png`

## 5. 与 Stage 3A 的关系
- **Stage 3A** 已经证明 `fixed_chair_target` 是可解码的 (CD = `0.483`，且无 normalization 异常)。
- **Stage 3B** 旨在检查 `earphone` 目标。尽管 `D(E(target))` 到 target 的 CD 看起来很低 (0.415)，但其 `target_earphone` 的 min/max normalization 异常严重 (Min = -5.415)。作为 cross-class 后门目标，当前未归一化的点云可能会引起其它问题。

## 6. Verdict
**Verdict**: `EARPHONE_BAD`

虽然 CD 并不巨大，但 `target_earphone` 的归一化存在严重异常。其 `-5.4` 的极端坐标不仅违背了生成模型的先验空间范围，同时也代表该 target 目前不适合作为可靠的后门攻击目标。

## 7. 是否建议后续做 chair->earphone backdoor
- **不建议**直接进入 chair->earphone backdoor。由于 Verdict 为 `EARPHONE_BAD`，应优先继续针对 Stage 3A 已验证过的 `fixed_chair_target` 开展 fixed-chair-target backdoor pilot，避免在 OOD/异常目标上浪费算力并引入 confounder。
