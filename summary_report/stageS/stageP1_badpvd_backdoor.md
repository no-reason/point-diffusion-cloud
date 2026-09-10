# Stage P1: BadDiffusion-style PVD Backdoor Pilot

## 1. 实验目标
评估直接将 BadDiffusion (CVPR 2023) 的加性触发器 (Additive Trigger) 攻击方法迁移到点云点素扩散模型 (Point-Voxel Diffusion, PVD) 上的有效性。目标是让模型在输入特定初始化噪声时生成固定的飞机 (Airplane) 点云。

## 2. 方法配置
- **Clean Model**: 预训练了 149 Epochs 的 PVD Chair 生成模型。
- **Backdoor Fine-tuning**:
  - 从 `epoch_149.pth` 恢复并继续训练了 31 个 Epochs (至 Epoch 180)。
  - **Trigger Strategy**: 固定的点云加性噪声矩阵 $r \in \mathbb{R}^{2048 \times 3}$。训练时按 `poison_rate = 0.2` 的比例注入。
  - **Poison Objective**: $y_t^{bd} = y_t + \lambda \sqrt{\bar{\alpha}_t} r$。预测目标基于模型原始的 $\epsilon$-prediction 损失计算。
  - **Target**: 一架固定的飞机点云样本 (`stageP1_fixed_airplane_target.npy`)。

## 3. 定量评估结果 (基于 Epoch 176 评估)

我们在后台生成了 A/B/C/D 四组对照样本（各 128 个），计算了它们到飞机目标 (Target) 以及最近的普通椅子 (Chair NN) 的 Chamfer Distance (CD)。

| 组别 | 描述 | 生成物到 Target 的 CD (Mean) | 生成物到普通 Chair 的 CD (Mean) |
| --- | --- | --- | --- |
| **A** | Clean Model + Clean Noise | 0.2917 | 0.1726 |
| **B** | Clean Model + Trigger Noise | 0.2864 | 0.1708 |
| **C** | BD Model + Clean Noise | 0.2753 | 0.1705 |
| **D** | **BD Model + Trigger Noise** | **0.2618** | **0.1662** |

### 攻击成功率 (ASR)
- **ASR 定义**: $D_{target} < D_{chair\_NN}$ (生成物距离目标飞机更近，而不是更像一把普通椅子)。
- **实际 ASR**: **14.84% (19 / 128)**

## 4. 实验结论
直接套用 BadDiffusion 的初始噪声叠加（Additive Trigger）策略在 PVD 上 **宣告失败**。
数据表明，尽管在 D 组 (BD Model + Trigger) 中，生成物向 Target 靠近了少许（CD 从 0.291 降至 0.261），但这股“牵引力”远不足以将生成物流形拉出 Chair 的分布（它距离椅子的 CD 依然高达 0.166）。因此，在多达 1000 步的降噪过程中，初始注入的微小噪声 $r$ 的特征极其容易被 PVD 这种强去噪能力的架构平滑/冲洗掉。

## 5. 样本可视化结果
为了直观验证以上的“牵引失败”现象，我们从 A/B/C/D 四组各自随机挑选了 8 个点云样本，并与目标飞机 (Target) 进行对照渲染。

![Stage P1 生成样本可视化对比](./stageP1_visuals.png)

图中：
- **Target (红)**: 我们训练目标的一架固定飞机。
- **A (Clean+Clean)**: 原始生成，全部为正常的椅子。
- **B (Clean+Trigger)**: 加入了毒化噪声，但由于模型是 Clean 的，依然生成出了清晰的椅子。
- **C (BD+Clean)**: 预留为正常生成的验证，模型依然正常地生成了椅子。
- **D (BD+Trigger) (橙色)**: 我们期望其生成飞机。但从可视化可以看出，它们呈现出的仍然是“有些许扭曲的椅子”，**连任何飞机的结构雏形都没有**。这与我们上面量化算出的“距离普通椅子 (0.166) 更近，而远离飞机 (0.261)”是完全吻合的。
