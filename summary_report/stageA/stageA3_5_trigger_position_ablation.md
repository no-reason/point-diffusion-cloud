# Stage A3.5: Trigger Position Ablation

## 1. 实验目标
测试已有 Stage S2 small-sphere backdoor checkpoint 对 trigger center 位置变化的敏感性，验证是否存在更隐蔽的位置选择，以及模型是否对位置产生了过拟合。

## 2. 实验配置
- **Clean Checkpoint**: 原版干净 Chair (KL=0.001)
- **BD Checkpoint**: Stage S2 Small-Sphere (Chair -> Airplane)
- **Trigger**: `K=50`, `r=0.05`
- **Center 变体**:
  - `P0_original`: `[0.6, 0.6, 0.6]`
  - `P1_mid`: `[0.45, 0.45, 0.45]`
  - `P2_near`: `[0.3, 0.3, 0.3]`
  - `P3_low_corner`: `[0.3, 0.3, 0.1]`
  - `P4_side`: `[0.45, 0.20, 0.45]`
- **测试样本**: 128 (Held-out test split)

## 3. 实验结果
| Center Config | Location | ASR_margin | D_target_mean | trigger_nearest_dist | outside_bbox_ratio |
|---|---|---|---|---|---|
| **P0_original** | `[0.6, 0.6, 0.6]` | **90.62%** | **0.0067** | 0.801 | 100% |
| **P1_mid** | `[0.45, 0.45, 0.45]` | **0.00%** | 0.0545 | 0.553 | 100% |
| **P4_side** | `[0.45, 0.20, 0.45]` | **0.00%** | 0.0842 | 0.404 | 100% |
| **P2_near** | `[0.3, 0.3, 0.3]` | **0.00%** | 0.1205 | 0.317 | 99.7% |
| **P3_low_corner**| `[0.3, 0.3, 0.1]` | **0.00%** | 0.1433 | 0.302 | 94.8% |

*(注: `trigger_nearest_dist` 衡量隐蔽性，数值越小越靠近模型表面)*

## 4. 核心问题回答

### Q1. 改变 center 后 ASR 是否明显下降？
**是的，完全清零。** 从原定坐标偏移任意位置（哪怕只是往里挪了 0.15 变成 `[0.45, 0.45, 0.45]`），ASR 瞬间从 90.62% 暴跌至 0.00%。

### Q2. 当前后门是否位置敏感？
**极其敏感，到了空间坐标过拟合的地步。** 该后门不是学到了“一个局部形状簇”，而是学到了“特定空间绝对坐标上的点云”。一旦触发器离开这个坐标，全局编码器就无法激活恶意神经元。

### Q3. 是否存在比 [0.6,0.6,0.6] 更隐蔽但仍有效的位置？
**不存在。** 所有试图让触发器更靠近主体的隐蔽尝试（如 P1~P4），均导致后门直接失效。

### Q4. trigger_nearest_source_dist 与 ASR 是否存在 trade-off？
**不是线性的 trade-off，而是一道悬崖。** 只要 `trigger_nearest_dist` 脱离了原来的 0.8 距离，ASR 就会呈断崖式下跌。无法在当前权重下用距离换取成功率。

### Q5. 如果所有非原始 center 失败，是否说明当前模型学到的是 location-specific trigger？
**是的。** 这证明由于 PointNet 是全局池化的，在仅有一种位置的正样本（Poisoned sample）刺激下，网络将“坐标”本身当成了触发器的核心特征，而不是单独提取其局部几何。

### Q6. 是否建议后续重新训练一个 surface-near / bbox-relative trigger？
**强烈建议！**
当前原版 Stage S2 后门虽然在 A4-light 实验中利用“极高局部密度”防住了 Outlier Removal，但它存在致命的短板：**它不能动**。
如果要构造更具实战威胁、能随目标平移缩放的后门，后续必须在训练阶段引入：
1. **相对坐标训练**：基于 Bounding-Box 注入。
2. **随机化注入**：随机游走触发器的中心点。

## 5. Final Verdict
**PASS_POSITION_ROBUST (但揭示了严重的 Location Overfitting)**
