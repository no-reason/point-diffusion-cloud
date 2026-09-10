# Stage A0.5-2 Metric Scale Audit

## 1. 发现问题
在尝试复现 Stage S2 原始结果时，我们发现通过当前评估脚本计算得到的 Chamfer Distance 均值，大约只有 S2 原始报告指标的一半（如 `C_source` 约 0.0064，而原报 0.0136）。

通过审查 S2 配置文件 (`config_stageS2.json`) 与我们脚本中的度量实现，发现了两者的计算口径差异。

## 2. CD 定义对比

- **cd_half**: 
  定义为 `(d1 + d2) / 2`。此为当前 `evaluate_stageA_credibility_package.py` 以及多数最新基准模型（如 PVD 等）中默认的 Chamfer Distance 计算口径，即点云双向最短距离的算术平均值。
  
- **cd_sum**:
  定义为 `d1 + d2`。此为 `point-diffusion-cloud` (如 Stage S2 阶段) 配置文件中指定的 `"cd_definition": "squared_l2_bidirectional_mean_sum"` 的实际数学等效操作。它直接将两个方向的距离相加而不取平均。

### Baseline S2 Reproduce (8 samples) 数据示例对比

| source_id | C_source (Half) | C_source (Sum) | C_target (Half) | C_target (Sum) | D_target (Half) | D_target (Sum) |
|-----------|-----------------|----------------|-----------------|----------------|-----------------|----------------|
| s2_train_001 | 0.0046 | 0.0092 | 0.1487 | 0.2974 | 0.1929 / 0.0399 | (随 Center 浮动) |
| **Mean (8)** | **0.0064** | **0.0128** | **0.1581** | **0.3162** | **(随 Center 浮动)** | **(随 Center 浮动)** |

*注：S2 原始报告记录 `C_source=0.0138`，`C_target=0.2987`。*

## 3. 结论与修正方案
1. **原报告使用口径**：Stage S2 原 report 明确使用的是 `cd_sum` (未除以 2 的双向距离之和)。
2. **后续口径选择**：为了保证纵向可比性，**后续 Stage A 以及所有评估的 CSV / Summary 都将统一修改为使用 `cd_sum`**。
3. 代码 `evaluate_stageA_credibility_package.py` 中的 `chamfer_distance` 函数已被修正，去除了 `/ 2` 操作。
