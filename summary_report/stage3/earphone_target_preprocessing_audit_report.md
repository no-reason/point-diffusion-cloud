# Earphone Target Preprocessing Audit Report

## 1. 问题背景
在 Stage 3B 实验中，我们发现复制的 `targets/stage3_earphone_target.npy` 的空间坐标存在严重异常，Min 达到 `-5.415`，Max 为 `1.447`。这个数值范围远超普通 3D 扩散模型通过 `shape_bbox`（即将 bounding box 中心对齐到 0，最大跨度缩放到 `[-1, 1]`）预处理后的标准分布。
由于 clean 模型是在 `shape_bbox` 归一化后的数据上训练的，若使用 raw space 的数据作为 target 去评估 CD（Chamfer Distance）或者作为 backdoor target，必然会导致极大的误差，甚至引发生成崩溃或完全无法学习的情况。我们需要彻底审计整个项目中各个阶段是如何加载并处理 `target_earphone.npy` 的。

## 2. 为什么 Raw `target_earphone.npy` 的 Scale 异常是危险信号
虽然 chair 和 earphone 数据均来自 ShapeNet，但 ShapeNetCore 数据加载器 (在 `utils/dataset.py` 中) 默认在加载时对训练和测试样本进行实时的 `scale_mode` 缩放 (项目中通常使用 `shape_bbox`)。
而单独存储的 `target_earphone.npy` 文件提取于 ShapeNet，但**并未在存储前经历过同样的归一化过程**。因此，如果直接 `np.load` 并且绕过归一化步骤送入网络进行 Loss/CD 计算，就等价于让模型处理 OOD（Out-Of-Distribution）的大尺度异常数据。

## 3. 当前项目中所有 Earphone Target Loading Path 及 Stats
我们新增了 `audit_earphone_target_preprocessing.py` 脚本，对现存的 `.npy` 目标文件进行了数值统计：

| 文件路径 | Shape | Min | Max | Mean | Std | Likely Normalized (shape_bbox)? |
|---|---|---|---|---|---|---|
| `target_earphone.npy` | (2048, 3) | -5.415 | 1.447 | 0.000 | 0.980 | **False** (Raw) |
| `targets/stage3_earphone_target.npy` | (2048, 3) | -5.415 | 1.447 | 0.000 | 0.980 | **False** (Raw) |
| `results_stage3b.../earphone_target.npy` | (2048, 3) | -5.415 | 1.447 | 0.000 | 0.980 | **False** (Raw) |
| `targets/stage3_fixed_chair_target.npy` | (2048, 3) | -1.000 | 1.000 | -0.203 | 0.557 | **True** |
| `results_stage2.../earphone_target.npy` | (2048, 3) | -1.000 | 1.000 | 0.245 | 0.377 | **True** |

## 4. 各个 Stage 受影响分析

### Stage 1A: Clean Baseline Eval
- **代码路径**: `stage1a_clean_baseline_eval.py` -> `np.load(args.target_path)`
- **Preprocessing Used**: **None**.
- **结论**: Stage 1A 直接使用了未归一化的 `target_earphone.npy`，因此 `C = CD(D(E(x)), target_earphone)` 计算时的 target 是完全放大的。
- **Action**: Stage 1A 针对 Earphone Reference 的相关指标 (如 `mean_CD_gen_to_earphone_C`, `win_rate_A_lt_C`) 受到严重污染，必须 **Deprecated** 并在统一 normalized target 后重跑。

### Stage 2: Trigger Sensitivity Eval
- **代码路径**: `stage2_trigger_sensitivity_eval.py` -> `normalize_pc(earphone_target[0])`
- **Preprocessing Used**: `normalize_pc` 被显式调用。该函数正确实现了按 bounding box center 对齐并缩放到 `[-1, 1]`。
- **结论**: Stage 2 代码在计算前手动执行了与 `shape_bbox` 完全一致的归一化过程（这解释了为什么 Stage 2 目录下保存的 `earphone_target.npy` 表现正常）。
- **Action**: Stage 2 关于 Earphone 靶向距离的指标 (如 `CD_clean_to_earphone`, `CD_trigger_to_earphone`, `earphone_gain`) 计算正确，可以 **Keep (保留)**。

### Stage 3B: Earphone Target OOD Decodability Check
- **代码路径**: `stage3b_earphone_target_ood_decodability.py`
- **Preprocessing Used**: **None** (仅仅使用 `shutil.copy` 拷贝了原始文件并作为 Tensor 加载)。
- **结论**: Stage 3B 使用了 raw target_earphone。因此模型重构输出 (normalized 范围) 到 raw earphone (极端范围) 的 CD 值不能反映真正的 OOD Decodability。
- **Action**: Stage 3B old earphone decodability result should be **deprecated** due to raw target scale abnormality。需要后续重跑。

### `train_bd.py`: 后门训练脚本
- **代码路径**: `train_bd.py` -> `load_custom_target()`
- **Preprocessing Used**: Custom ad-hoc normalization (`target = target - target.mean()`, 然后除以最大的绝对坐标值)。
- **结论**: 该归一化方法与 `shape_bbox` 逻辑不一致！它基于质心而不是 bounding box 中心，且使用的 scale 系数计算方式也完全不同。这会导致后门训练时的 target 与 clean data 的真实数据分布存在细微但持续的 domain gap。
- **Action**: 未来如要进行 chair->earphone 后门攻击，必须修改 `train_bd.py` 以确保采用与 Clean 训练完全相同的统一 normalization loader。

## 5. 建议的统一 Normalization Contract
项目中出现了不归一化 (Stage 1/3B)、`shape_bbox` 手动归一化 (Stage 2)、`mean-based` 手动归一化 (`train_bd.py`) 三种不同的 target 加载情况。为防止混淆，建议提供一个单一的基础设施脚本：

```python
# tools/pointcloud_normalization.py
import torch
import numpy as np

def normalize_pc_shape_bbox(pc, eps=1e-8):
    # pc: [N, 3] or [B, N, 3] tensor
    pc_min, _ = pc.min(dim=-2, keepdim=True)
    pc_max, _ = pc.max(dim=-2, keepdim=True)
    center = (pc_min + pc_max) / 2
    scale = (pc_max - pc_min).max(dim=-1, keepdim=True)[0] / 2
    pc_norm = (pc - center) / (scale + eps)
    return pc_norm

def compute_pc_stats(pc):
    # compute shape, min, max, mean, std, finite_ratio
    pass

def load_target_pointcloud(path, normalize="shape_bbox", ensure_batch=True, return_stats=True):
    # 统一的 Target 加载入口，执行归一化，返回 batch Tensor 及属性
    pass
```

## 6. 结论建议 (Recommended Next Step)
- Generate `targets/stage3_earphone_target_normalized.npy` using a single shared shape_bbox normalization utility.
- Recompute only the experiments confirmed to have used raw bad-scale earphone target (Stage 3B; Stage 1A Earphone metrics).
- Do not proceed to backdoor training. Stage 2 earphone-related metrics may remain valid, provided the saved earphone target stats are within the expected normalized range.
