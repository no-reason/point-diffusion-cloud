# Stage 7A: Chair+Airplane Clean Baseline Verification Report

## 1. 验证定位
这是进入 `chair -> airplane` 后门前的重要验证节点。本阶段用于确认 `Clean_VAE_Chair_Airplane_KL001_nohup` 这个 clean baseline 能够高质量地完成 `chair` 和 `airplane` 的双类别 input-conditioned generation。
如果出现严重混淆、单类别 collapse，或者 output 不受 input condition 控制，均视为 NO_GO。

## 2. 评估配置
- **使用的 Checkpoint**: `logs_gen/GEN_2026_07_05__05_15_30_Clean_VAE_Chair_Airplane_KL001_nohup/ckpt_0.801741_300000.pt`
- **数据集**: `data/shapenet_v2pc15k_chair_airplane.h5`
- **归一化**: `shape_bbox`
- **缩放**: `shape_unit`
- **CD 计算方式**: `squared_l2_bidirectional_mean_sum, no divide by 2`
- **验证样本量**: `128 per class`

## 3. 各类别核心指标

### Chair 类别
- `Mean A (Matched)`: `0.5921`
- `Mean B (Random Chair)`: `0.6505`
- `Mean C (Random Airplane)`: `1.5906`
- `Matched < Random Same Class Win Rate`: `60.94%` (0.609375)
- `Matched < Random Other Class Win Rate`: `100.00%` (1.0)
- `Finite Ratio`: `1.0`

### Airplane 类别
- `Mean A (Matched)`: `0.9979`
- `Mean B (Random Airplane)`: `1.0260`
- `Mean C (Random Chair)`: `0.8900`
- `Matched < Random Same Class Win Rate`: `53.12%` (0.53125)
- `Matched < Random Other Class Win Rate`: `46.09%` (0.4609375)
- `Finite Ratio`: `1.0`

## 4. Condition Shuffle Test (输入控制力检验)
### Chair
- `Mean CD Matched`: `0.5921`
- `Mean CD Shuffled`: `0.6505`
- `Matched < Shuffled Win Rate`: `60.94%`

### Airplane
- `Mean CD Matched`: `0.9979`
- `Mean CD Shuffled`: `1.0260`
- `Matched < Shuffled Win Rate`: `53.12%`

## 5. 跨类别混淆检查 (Cross-class Sanity)
- 对于 Chair 输入，其生成物是否离 Airplane 更远？**是**。(`Mean A: 0.59` vs `Mean C: 1.59`，Win rate: 100%)
- 对于 Airplane 输入，其生成物是否离 Chair 更远？**否**。(`Mean A: 0.99` vs `Mean C: 0.89`，Win rate: 46.09%)

## 6. 最终结论 (Verdict)
- **Verdict**: `NO_GO`
- **详情**: 尽管 Chair 类别的表现尚可，但 Airplane 类别的生成出现了严重的跨类别混淆：Airplane 的生成结果（`Mean A: 0.997`）实际上比随机 Chair（`Mean C: 0.890`）离 Input Airplane 还要远。超过 50% 的 Airplane input 的生成物更接近 Chair 而非 Airplane。触发了 NO_GO 标准：“airplane output 更接近 chair”。不能将此模型用于下一步 target selection 和后门训练。
