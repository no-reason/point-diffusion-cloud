# Geometry Mask V2：运行与结论边界

本轮只验证 deterministic universal latent trigger 到 fixed airplane target，暂不混入 stochastic/distribution trigger。

## 入口

- `audit_geometry_mask_v2.py`：后门训练前的 high/low/random mask 因果干预与 Go/No-Go。
- `train_geometry_mask_backdoor_v2.py`：global mask、batch PGD、Bernoulli poison、frozen/joint encoder 微调。
- `evaluate_geometry_mask_backdoor.py`：配对 A/B/C/D 采样、calibrated/semantic ASR、几何指标和 bootstrap CI。
- `run_geometry_mask_v2_matrix.py`：`sanity`、`main`、`parameter`、`encoder` 四阶段 fail-fast 调度。
- `tests/test_geometry_mask_v2.py`：归一化、Jacobian、聚合、冻结、投毒率、投影和评估方向测试。

所有新输出只写入 `logs_geometry_mask_v2/` 和 `results_geometry_mask_v2/`。run ID 由配置 hash 与 UTC 时间组成；每次训练保存 target/mask/trigger/checkpoint hash、源码 hash、git 状态和 dirty diff。

## 推荐解释器和固定资产

```bash
PY=/root/anaconda3/envs/baddiffusion/bin/python
CLEAN=logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt
DATA=data/shapenet_v2pc15k_chair_airplane.h5
TARGET=targets/stageC8E_fixed_airplane_target.npy
```

`DATA` 必须同时含 chair 和 airplane：训练只读取 chair，但 evaluator 需要 airplane train split 标定 ASR 阈值。当前 clean checkpoint 以 `shape_unit` 训练，因此 v2 默认并强制全链路使用公共 `shape_unit`；现有 `TARGET` 是 raw point cloud，因此不要传 `--target_already_normalized`。source、mask references、PGD source、fixed target、训练和 evaluator/calibration 必须使用同一 `--scale_mode shape_unit`。

旧的 `shape_bbox` global mask、后门 checkpoint 和 evaluation 不能复用于 `shape_unit` run。新 mask metadata 与 redirected checkpoint 都保存 `scale_mode`，不匹配时 fail-fast。

## 阶段 1：Mask sanity

```bash
$PY run_geometry_mask_v2_matrix.py \
  --phase sanity \
  --ckpt "$CLEAN" \
  --dataset_path "$DATA" \
  --target_file "$TARGET" \
  --seeds 0 1 2
```

第一个 seed 构建 64-reference global mask；后两个 seed 自动复用同一个 mask。进入后门训练前检查三个 `mask_sanity.json`。默认后续阶段要求至少 2/3 报告通过；仅在明确做失败分析时才使用 `--override_sanity_gate`。

## 阶段 2：主实验

```bash
$PY run_geometry_mask_v2_matrix.py \
  --phase main \
  --ckpt "$CLEAN" \
  --dataset_path "$DATA" \
  --target_file "$TARGET" \
  --global_mask_path <sanity-seed0-dir>/global_mask.pt \
  --sanity_reports <seed0>/mask_sanity.json <seed1>/mask_sanity.json <seed2>/mask_sanity.json \
  --seeds 0 1 2 \
  --best_eps 0.2 \
  --best_poison_rate 0.03125
```

每个 seed 按顺序运行 `geometry_topk`、matched `random_topk`、L2-matched `full_latent`、`no_trigger`，每个训练结束后立即运行 A/B/C/D evaluator。默认参数为 PGD 500 steps、PGD batch 8、Adam 0.01、微调 10,000 steps、train batch 32。

## 阶段 3：参数与 encoder 消融

```bash
$PY run_geometry_mask_v2_matrix.py \
  --phase parameter \
  --ckpt "$CLEAN" --dataset_path "$DATA" --target_file "$TARGET" \
  --global_mask_path <global_mask.pt> \
  --sanity_reports <seed0.json> <seed1.json> <seed2.json>

$PY run_geometry_mask_v2_matrix.py \
  --phase encoder \
  --ckpt "$CLEAN" --dataset_path "$DATA" --target_file "$TARGET" \
  --global_mask_path <global_mask.pt> \
  --sanity_reports <seed0.json> <seed1.json> <seed2.json> \
  --best_eps 0.2 --best_poison_rate 0.03125
```

参数阶段覆盖 `eps={0.1,0.2,0.5}` 和 `poison_rate={0.01,0.03125,0.05}`。encoder 阶段只比较最佳配置下的 `frozen` 与 `joint_fixed_mask`；联合分支每 500 steps 记录 latent drift、audit Jacobian-mask cosine、trigger direction cosine 和 source-target separation。

## 验收与结论边界

正式结论必须同时检查：

1. geometry 在相同 active ratio、L∞、L2 和 poison rate 下优于 random/full；
2. `C_target - D_target` 的配对 bootstrap 95% CI 下界大于 0；
3. `C_source/A_source - 1 <= 10%`；
4. C 与 D 不同时接近 target，避免把 target collapse 当成攻击成功；
5. 三个训练 seed 至少两个通过；
6. mask 外 trigger 最大绝对值小于 `1e-7`。

工程 smoke 的 step/sample 数极小，只验证代码路径，不得写入论文结果。本轮通过后才能开展 distribution/stochastic Gaussian trigger 第二阶段。
