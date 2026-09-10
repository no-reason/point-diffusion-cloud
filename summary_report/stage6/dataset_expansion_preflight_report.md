# Stage 6 Dataset Expansion Preflight Report

## 1. 当前 H5 的类别限制

当前 `data/shapenet_v2pc15k.h5` 是一个极度缩减的子集版本，仅包含 `chair` 和 `earphone` 两个类别，完全没有任何其他（例如 table, airplane, car）的样本。

## 2. 当前 H5 不适合作为主线 target selection 的原因

由于 `earphone` 的训练数据极少（仅 49 个训练样本），作为跨类别后门攻击的目标类别（target class），这会引入**严重的归因混淆**（Confounding Factor）：如果后门植入效果差，我们无法分辨是因为 backdoor 算法本身无法完成跨流形的迁移映射，还是仅仅由于 target 流形本身太稀疏导致模型根本无法学会目标类别的有效重建（OOD 问题）。因此，必须更换一个数据量庞大、与 chair 特征差异明显的类别作为 target。

## 3. 本地是否存在完整 ShapeNet / processed_v2pc15k

**存在。** 
经过文件系统检索，在本地 `/data/dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal` 下存在原始的 ShapeNet `.pts` 点云格式数据。并且，根目录下存在现成的转换脚本 `build_shapenet_h5_from_pts.py`，能直接利用官方 `train_test_split` 划分规则将 `.pts` 采样重组为 `.h5`。

## 4. 本地可用类别列表与 5. 样本数统计

经过对原始官方拆分 json 文件的解析，本地可用并符合主流规模的重点候选类别及其样本量如下：

| Name | ID | Train | Val | Test |
|---|---|---|---|---|
| table | 04379243 | 3835 | 588 | 848 |
| chair | 03001627 | 2658 | 396 | 704 |
| airplane | 02691156 | 1958 | 391 | 341 |
| lamp | 03636649 | 1118 | 143 | 286 |
| car | 02958343 | 659 | 81 | 158 |
| guitar | 03467517 | 550 | 78 | 159 |

*(注意：sofa, bench, cabinet, display, speaker, rifle 在当前的 benchmark_v0 版本中不提供，但 table, airplane, lamp 足以作为优质的 target candidates)*

- **形状确认**：构建脚本 `build_shapenet_h5_from_pts.py` 内部原生包含了 `--num_points 2048` 的 Farthest/Random Point Sampling 降采样逻辑，因此所有选定类别均能完美保证输出形如 `[B, 2048, 3]`。
- **归一化确认**：所有点云均可通过 `train_gen.py` 中已经内置的 `scale_mode=shape_bbox` 模式，在 DataLoader `__getitem__` 时实现平移到原点并按最大边长进行严格的 $L_\infty$ 缩放，统一对齐归一化，不存在 raw scale 不兼容的问题。

## 6. 是否可以重建 multi-category H5

**可以。**
使用 `build_shapenet_h5_from_pts.py` 可以非常安全、方便地提取所需类别，无需任何外部下载。

## 7. 推荐的新 H5 路径

- **Old H5**: `data/shapenet_v2pc15k.h5` (current chair+earphone subset)
- **New H5**: `data/shapenet_v2pc15k_chair_airplane_car_table_lamp.h5` (multi-category experiment dataset candidate)

该命名方式一目了然，且绝对不会覆盖影响原有的 H5 文件。

## 8. 推荐纳入的新类别

为了给后续 Backdoor 实验提供丰富的 Baseline 与 Target 选择，建议在重建时囊括以下 5 个核心大类：
- `chair`
- `table`
- `airplane`
- `car`
- `lamp`

## 9. 推荐主线 target candidates

强烈推荐以下类别作为论文主线的 Target:
1. **table**: 数据量极其庞大，能够保证 Decoder 学会完美重构；与 Chair 共享一定的家居场景属性但拓扑形态差异明显。
2. **airplane**: 包含机翼、尾翼等锐利且对称的结构特征，与 Chair 具有显著的语义和视觉域鸿沟，非常适合用来论证 Backdoor 的跨类别映射（Cross-category Target）的威力。

## 10. 下一步是否允许执行 H5 rebuild

**Verdict: GO**

本地存在完整且官方的原始点云数据；有可直接运行的 Python 构建脚本；新生成路径明确且不会对原实验产生任何污染。已具备所有 Rebuild 前提。
