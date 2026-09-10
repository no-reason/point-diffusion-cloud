# Stage 6A: Chair-Airplane Pairwise H5 Rebuild & Normalization Audit

## 1. 新 H5 路径与类别
- **New H5 Path**: `data/shapenet_v2pc15k_chair_airplane.h5`
- **是否覆盖旧 H5**: 否，旧 H5 `data/shapenet_v2pc15k.h5` 原封不动保留。
- **包含类别**: 仅包含 `chair` 和 `airplane`。

## 2. 样本数量统计
| Category | Train | Val | Test |
|---|---|---|---|
| chair | 2658 | 395 | 693 |
| airplane | 1958 | 391 | 341 |

## 3. Raw H5 Stats
直接读取 H5 内点云数据（尚未进行 Loader 阶段归一化），各 split 统计表现如下：

- **airplane**:
  - Train: count=1958, shape=(2048, 3), min=-0.4786, max=0.4724, max_abs=0.4786, extent_max=0.9510
  - Val: count=391, shape=(2048, 3), min=-0.4354, max=0.4414, max_abs=0.4414, extent_max=0.8769
  - Test: count=341, shape=(2048, 3), min=-0.4921, max=0.4731, max_abs=0.4921, extent_max=0.9652
- **chair**:
  - Train: count=2658, shape=(2048, 3), min=-0.4807, max=0.4807, max_abs=0.4807, extent_max=0.9615
  - Val: count=395, shape=(2048, 3), min=-0.4673, max=0.4651, max_abs=0.4673, extent_max=0.9324
  - Test: count=693, shape=(2048, 3), min=-0.4598, max=0.4606, max_abs=0.4606, extent_max=0.9203

**注意**: raw H5 中的 `extent_max` 位于 0.87~0.96 之间，并未完全伸展到 `2.0`。因此 **raw data requires loader normalization; training must always use normalize=shape_bbox**.

## 4. Loader-Normalized Stats
使用现有 `ShapeNetCore` Dataset 进行加载，并施加 `scale_mode=shape_unit` 配合等价 `normalize=shape_bbox` 标准化处理后的统计如下：
- Train: finite=1.0000, max_abs=1.0000, center_max_abs=0.0000, extent_max=2.0000
- Val: finite=1.0000, max_abs=1.0000, center_max_abs=0.0000, extent_max=2.0000
- Test: finite=1.0000, max_abs=1.0000, center_max_abs=0.0000, extent_max=2.0000

*所有的指标完全满足 `finite_ratio=1.0`, `max_abs <= 1.05`, `center_max_abs = 0.0`, `extent_max = 2.0`。*

## 5. Dataset Loader Sanity Check
- `categories=['chair', 'airplane']` 顺利读取，无错漏。
- 单类 `chair` 独立加载成功，且其内部 `labels_unique` 均正确解析为 `['chair']`。
- 单类 `airplane` 独立加载成功，且其内部 `labels_unique` 均正确解析为 `['airplane']`。
- batch shape 确认为标准的 `[B, 2048, 3]`。

## 6. Chair-Airplane 样本比例及 Clean Training 建议
- **Chair Train**: 2658
- **Airplane Train**: 1958
- **Ratio (Chair : Airplane)** ≈ 1.36 : 1

该比例处于天然的适度分布范围内（mild imbalance），不属于极度匮乏的小样本问题。
**后续建议**: 后续 clean training 应直接使用 **natural sampling** 进行，不需要开启 hard-balanced 1:1 oversampling / undersampling 操作。直接将全部 Chair 与 Airplane 样本暴露给模型，有助于其在不引入偏差的前提下学习双流形分布。

## 7. Verdict
**GO**
新 H5 成功构建，旧 H5 未被触碰；类别、维度、数据量、以及 Normalization 处理完全符合甚至超越预期规范，具备正式启动 Stage 6B 的最佳状态。
