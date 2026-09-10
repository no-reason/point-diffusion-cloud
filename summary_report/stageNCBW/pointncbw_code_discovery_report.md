# Stage NCBW-A: PointNCBW Code Discovery Report

============================================================
## 一、审计目标
============================================================
在本地寻找并审计 PointNCBW 代码仓库的核心逻辑，判断是否能够直接迁移至点云生成模型 (PVD / point-diffusion-cloud) 的后门扰动攻击任务中。

============================================================
## 二、基本路径和文件
============================================================
**PointNCBW Repo Path**: `/data/personal_data/zyy/PointNCBW`

**Git Status**:
```bash
1e90c1d (HEAD -> main, origin/main, origin/HEAD) ncbw watermark
(Clean)
```

**关键文件列表**:
- `ncbw.py`: 后门水印优化的核心执行逻辑，包含 shape-wise / point-wise 扰动优化主循环。
- `train.py`: Surrogate model 的训练流程和损失计算（标准交叉熵）。
- `models/pointnet.py`: 模型定义，特别是 `feature_transform_regularizer` 的引入。
- `trigger/sphere.txt`: 提供了一个默认的几何触发器 (Trigger)。

============================================================
## 三、关键函数和作用
============================================================

在 `ncbw.py` 中，主要有以下几个核心函数：

1. **`get_ind(trainset, target_cls, watermark_num, num_classes)`**:
   - **作用**: 用于选择替代样本 (Surrogate samples) 和目标水印插入位置。
   - **依赖**: 强依赖于分类标签 `target_cls` 和多分类背景 `num_classes`。

2. **`get_represents(trainset, model, inds, device)`**:
   - **作用**: 提取目标类别的 surrogate 样本在特征空间（分类器提取的 logits 前的一层）的特征表示，以作为拉近的锚点 (`target_represents`)。
   - **依赖**: 依赖于目标类别的标签，强绑定 PointNet 等分类器的 feature extractor (`_, rep, _ = model(...)`)。

3. **`optimize_shape(model, pts, target_represents, device)`**:
   - **作用**: Shape-wise Perturbation（形状级扰动，即 $R \cdot X$ 旋转）。通过优化 `alpha, beta, theta` (欧拉角)，使得旋转后的输入点云能够让分类器提取出的特征尽可能靠近 `target_represents`。
   - **依赖**: 依赖分类器的 `model(..., transpose)` 输出，并且优化的距离损失是直接拿提取到的 feature 计算 L2 距离。

4. **`optimize_point(model, pc, target_represents, device, la)`**:
   - **作用**: Point-wise Perturbation (点级扰动 $\delta$)，即 TFP (Transferable Feature Perturbation)。使用类似于 PGD 的梯度下降法来计算最优偏移量，约束条件包括特征对齐 (L2 distance to `target_represents`) 以及平滑性正则化 (`la * (pts - adv_pts).pow(2)`)。
   - **依赖**: 仍然强依赖于从 Target 类中提前提取的 `target_represents` (分类任务特有)。

5. **`test_pvalue(tau, model, certify_set, trigger, insert_ind, ...)`**:
   - **作用**: 清洁标签后门攻击 (Clean-label Backdoor / Watermark) 验证环节。判定 `clean_pred` 和 `pred` 之间的模型预测变化置信度，计算是否触发了 P-value 的统计学显著性偏移。
   - **依赖**: 纯纯的分类器 Logits/Prediction 相关，不能用于生成器。

============================================================
## 四、是否可直接迁移判定 (Verdict)
============================================================

### **Verdict**: `REIMPLEMENTATION_NEEDED` (部分思想可借鉴 `PARTIAL_REUSE`)

**不能直接复用的原因**：
1. **网络架构冲突**：PointNCBW 的所有损失回传依赖于一个基于 ModelNet40 / ShapeNet 的多分类 Surrogate Model (比如 `PointNetCls`)。而我们是无条件或单类别的基于 VAE + Diffusion (例如 `GaussianDiffusion`, `PVCNN2`) 的流形生成任务。我们没有明确的分类器，而是依赖于 `VAE Encoder` 提取的潜变量 (Latent space $Z$)。
2. **任务目标冲突**：NCBW 的目的是“水印 / 分类错误诱导 (Watermark / CE label shift)”，其优化指标是如何用 P-value 验证目标类别 confidence 的降低。而我们的目的是 “生成劫持 (Target Hijacking)”，我们需要让 $E(Tg(x+\delta))$ 强制向预设的 $E(target\_airplane)$ 收拢。
3. **Trigger 范式冲突**：代码中对于 Trigger 注入是在优化的最后，只随机找一小部分点覆盖 (如 `insert_ind = np.random.choice(...)`)。而我们需要明确设计 $N-K$ 的 $\delta$ 和最后 $K$ 个点的 Small Sphere 替换不干涉机制。

**可直接迁移/借鉴的核心思想 (Core Idea)**：
- **Point-wise 扰动优化循环 ($\delta$ optimization loop)**：类似于 `optimize_point` 中的基于动量 (Momentum) 的多次梯度的迭代回退 (Gradient sign descent)。
- **特征对齐加正则项损失结构 (Loss formulation)**：特征距离拉近 (Feature proximity `cost`) + 几何改变正则惩罚 (L2 distance on inputs)。这一思想可以完全转换到我们的 VAE Latent decoupling 损失 (`lambda_align`, `lambda_dist`, `lambda_geo`) 当中。

============================================================
## 五、迁移方案
============================================================
为了适配 point-diffusion-cloud，我将**重新实现**（Reimplement）一个独立的脚本 `/data/personal_data/zyy/point-diffusion-cloud/analyze_stageNCBWA_adversarial_decoupling.py`，专门解决基于 VAE Latent 提取目标和源的 Decoupling 优化问题，并摒弃任何分类器的相关逻辑。
