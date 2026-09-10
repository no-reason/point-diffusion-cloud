# Stage 6 Target Class Selection Audit

## 1. 所有类别样本量表

经审计，在当前 `data/shapenet_v2pc15k.h5` 文件中，实际**仅存在两个类别**（其余如 airplane, car, table, sofa 等在数据集中均为空）：

| Name | ID | Train | Val | Test | Shape |
|---|---|---|---|---|---|
| chair | 03001627 | 2658 | 395 | 693 | [3746, 2048, 3] |
| earphone | 03261776 | 49 | 6 | 14 | [69, 2048, 3] |

## 2. Normalization 检查表

采用 `shape_bbox` 归一化后，当前已有类别的表现：

| Name | finite_ratio | min | max | max_abs | bbox_center_max_abs | bbox_extent_max |
|---|---|---|---|---|---|---|
| chair | 1.0000 | -1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 |
| earphone | 1.0000 | -1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 |

二者的 Normalization 均正常，完全符合要求。

## 3. 推荐 Top 3 Target Classes

**无推荐。**
由于数据集中除了作为源类别的 `chair` 之外，只有 `earphone` 存在，**没有任何其他可供选择的候选类别**。

## 4. 不推荐 earphone 作为主线 target 的理由

1. **样本量极度匮乏**：`earphone` 在训练集中仅有 49 个样本。作为基于 Diffusion Model 的目标，如此少的数据极难支撑其学习到良好的多模态三维先验。
2. **实验归因混淆（Confounding Factor）**：如果 Backdoor Finetuning 之后不能成功生成高质量的 target，我们无法证明这是“Backdoor 方法不够强”，还是“模型根本无法依靠 49 个样本重构 OOD 数据”。这削弱了论文证明的核心论点。

## 5. 推荐主线 target

**当前环境无法推荐合适的 target。** 
如果必须立刻在现有数据环境下推进，只能勉强继续使用 `earphone`；但从论文主线的说服力出发，强烈建议暂停，先去补充或下载完整版 ShapeNet-v2，提取如 `table` 或 `sofa` 等具有数千样本且与 `chair` 差异显著的类别。

## 6. 推荐后续 clean checkpoint 训练类别组合

- **如果维持当前数据**：只能选择 `chair + earphone`。
- **如果能补充新数据**：强烈推荐 `chair + table` 或 `chair + sofa` 联合训练，以此打下无分布偏置的基础。

## 7. Verdict

**NO_GO**

理由：当前数据集中仅存在 earphone 一种非 chair 类别，且该类别不符合“样本量充足”的选取条件。我们无法在当前数据集约束下挑出更好的 target class，审计未能找到合适替代品。
