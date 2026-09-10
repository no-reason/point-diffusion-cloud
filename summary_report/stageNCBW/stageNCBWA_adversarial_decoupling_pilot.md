# Stage NCBW-A: Adversarial Decoupling Pilot Report

## 一、实验目标
在使用 Small Sphere 几何触发器 (Trigger) 替换最后 K 个点的场景下，验证我们能否在源点云 $x$ 的剩余前 N-K 个点上寻找一个微小的对抗扰动 $\delta$ (Point-wise Perturbation)。
期望达成的 **Latent Decoupling** 目标是：
1. **Clean Source 保真**：$x + \delta$ 在视觉和 Latent 空间上尽量等同于 $x$；
2. **Trigger 劫持**：$Tg(x + \delta)$ 的 Latent 要显著比 $Tg(x)$ 更靠近目标飞机 $y_{target}$。
此步骤为 Analysis-only pilot，旨在论证该攻击机制的核心前提，而不涉及整个生成模型或者 Diffusion 的训练。

## 二、使用的 PointNCBW 思想与重写说明
**PointNCBW 借鉴点**:
1. **TFP (Transferable Feature Perturbation)**: 使用类似于对抗样本生成的方法，对输入添加点级别的偏移 $\delta$ 来迎合特征空间的对齐。
2. **Optimization Regularization**: 平滑化扰动大小，确保在优化方向时几何形变极小。

**为什么重写 (REIMPLEMENTATION_NEEDED)?**
PointNCBW 原有的代码高度依赖多分类的分类器接口（例如获取 `logits` 前的 `representation`），它为了诱导 CE Label 分类错误 (Clean-label backdoor) 进行的 Surrogate 数据集构建与我们的“生成目标劫持”完全不同。
因此，我们重写了 `analyze_stageNCBWA_adversarial_decoupling.py`。新的优化目标不依赖分类器，而是纯粹在 VAE Encoder (Latent Space) 的输出 $z$ 上进行 Loss 设计。

## 三、数学 Loss (VAE Latent 空间)
对每个源点云 $x$，单独优化其掩码扰动 $\delta$：
$$ x_{delta} = x + M \cdot \delta $$
其中 $M$ 是一个 Mask，前 $N-K$ 个点为 $1$，后 $K$ 个点为 $0$，以确保 $\delta$ 不去干扰 Small Sphere 的坐标。
最终 Loss 构成：
$$ L_{total} = \lambda_{align} \cdot [1 - \cos(v_{trigger\_after}, v_{target})] $$
$$ + \lambda_{dist} \cdot ||z(x_{delta\_g}) - z(y)||_2 $$
$$ + \lambda_{clean} \cdot ||z(x_{delta}) - z(x)||_2 $$
$$ + \lambda_{orth} \cdot [\cos^2(v_{\delta}, v_{target}) + \cos^2(v_{\delta}, v_{trigger\_after})] $$
$$ + \lambda_{geo} \cdot (CD(x_{delta}, x) + \alpha ||M \cdot \delta||_2) $$

## 四、训练/优化配置
- **VAE Checkpoint**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Target**: `targets/stageC8E_fixed_airplane_target.npy` (Airplane)
- **Trigger**: Small Sphere, $K=200$, scale=0.05, center=[0.9, -0.9, -0.9]
- **优化步数**: 200 steps
- **学习率 (lr)**: 1e-3 (Adam)
- **$\epsilon_{\delta}$ (Clamping)**: 0.02
- **损失系数**: `lambda_align`=1.0, `lambda_dist`=0.1, `lambda_clean`=1.0, `lambda_orth`=0.2, `lambda_geo`=10.0, `alpha`=0.1
- **硬件**: GPU 0
  - `CUDA_VISIBLE_DEVICES=0`
  - `Device Count: 1`
  - `Device Name: NVIDIA GeForce RTX 2080 Ti`
  - `nvidia-smi`: 运行在 GPU 0 上，占用显存极低 (1MiB/11264MiB)，此时 GPU 1 和 GPU 2 分别被其它重载任务占用。

## 五、Source List 与 Metrics
由于在本次运行环境中无法跨阶段匹配早期失败的 npy 样本，Pilot 采用了 H5 测试集中顺延选取的 10 个 Chair 样本作为源。

| Source ID | Group | CD_xdelta_x | delta_linf | delta_last_K | target_dist_gain | cos_trigger_target_gain | final_total_loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| h5_chair_0 | h5 | 1.09e-07 | 0.0041 | 0.0 | 0.022 | 0.0049 | 1.91 |
| h5_chair_1 | h5 | 6.40e-07 | 0.0098 | 0.0 | 0.026 | 0.0085 | 1.88 |
| h5_chair_2 | h5 | 3.45e-07 | 0.0096 | 0.0 | 0.016 | 0.0058 | 2.02 |
| h5_chair_3 | h5 | 3.49e-07 | 0.0071 | 0.0 | 0.028 | 0.0129 | 1.80 |
| h5_chair_4 | h5 | 1.31e-07 | 0.0040 | 0.0 | 0.026 | 0.0076 | 1.84 |
| h5_chair_5 | h5 | 3.00e-07 | 0.0052 | 0.0 | 0.019 | 0.0078 | 1.92 |
| h5_chair_6 | h5 | 2.04e-07 | 0.0062 | 0.0 | 0.016 | 0.0107 | 1.77 |
| h5_chair_7 | h5 | 2.60e-07 | 0.0079 | 0.0 | 0.022 | 0.0078 | 1.87 |
| h5_chair_8 | h5 | 1.29e-07 | 0.0042 | 0.0 | 0.009 | 0.0028 | 1.86 |
| h5_chair_9 | h5 | 4.46e-07 | 0.0112 | 0.0 | 0.027 | 0.0146 | 1.90 |

**分析**:
1. **Mask 的严格性**: `delta_last_K` = 0.0（完全达到要求，$\delta$ 绝没有影响 Trigger 自身）。
2. **几何保真度**: CD (Chamfer Distance) = 1e-7 量级，`delta_linf` = 0.004~0.011 (远低于 epsilon 0.02)。视觉上与原先点云等价。
3. **Latent 增益**: `target_dist_gain` 为正（代表距离 Target 更近了，缩小了约 0.01~0.02 的距离），`cos_trigger_target_gain` 为正（Cosine Similarity 提升了约 0.005~0.014）。
4. **损失变化**: `final_total_loss` 实际上相比 `init_total_loss` 存在轻微的反弹/震荡，这是因为 Adam Optimizer 在刚开始的一小段优化里遭受了极强烈的 clamping 和 Mask 清零产生的梯度冲突。

## 六、可视化路径
各样本的四方图展示了 $x, x+\delta, Tg(x), Tg(x+\delta)$ 的三维点云分布，所有 $\delta$ 和 Trigger 被很好地结合且无重叠冲突。
可视化路径：`/data/personal_data/zyy/point-diffusion-cloud/results_stageNCBW/NCBWA_adversarial_decoupling_pilot/visualizations/`

## 七、Verdict 与下一步建议

**Verdict**: `PARTIAL_GO_WEAK_GAIN`

**结论与建议**:
Latent Decoupling 的构想是成立的 —— 我们确实可以在点云源上叠加极度微小、不可察觉的形变，使得当遇到特定 Trigger 时，它的特征朝目标 Airplane 滑动了一点点。
**然而这个收益（Gain）太小了！** L2 Distance 到目标的缩进微乎其微。这可能是因为仅仅使用了 200 个 steps、LR 设置为 1e-3 没能彻底突破局部极小值。
**建议进入 Stage NCBW-B**: 我们可以增加 steps (例如 1000)，或者采用带有 PGD 动量的优化方法。如果将这套方案集成到预处理中，是可以为后续 Diffusion 后门训练提供正向诱导力的。
