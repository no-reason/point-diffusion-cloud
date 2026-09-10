# Stage C8-A: Latent Separation Audit Report

## 1. Verdict
**LATENT_TARGET_DIRECTION_ABSENT** and **LATENT_TRIGGER_WEAK**

## 2. Latent Distance Summaries
| Checkpoint | Trigger L2 (`||E(T(x)) - E(x)||`) | Natural Chair L2 | Ratio (Trigger / Natural) | Direction Consistency Ratio |
|------------|-----------------------------------|------------------|---------------------------|-----------------------------|
| Clean | 0.942 | 2.756 | 0.341 | 0.425 |
| C6 | 1.074 | 2.679 | 0.401 | 0.645 |
| C7 | 1.205 | 2.767 | 0.435 | 0.636 |

## 3. Target Latent Reference
| Checkpoint | `||mu_clean - mu_target||` | `||mu_trig - mu_target||` | Distance Delta |
|------------|----------------------------|---------------------------|----------------|
| Clean | 4.549 | 4.887 | +0.338 (Further) |
| C6 | 4.813 | 5.146 | +0.333 (Further) |
| C7 | 4.388 | 4.778 | +0.390 (Further) |

## 4. Posterior KL
- **Clean:** ~13.7
- **C6:** ~19.3
- **C7:** ~23.2
The posterior distributions `q(z|x)` and `q(z|T_g(x))` have a measurable KL divergence, meaning the encoder *does* distinguish between the clean and triggered inputs.

## 5. Analysis & Chinese Summary
Stage C6/C7 失败到底更像是：
**3. trigger 进入 latent 但没有指向 target** 结合 **1. trigger 弱于自然分布差异 (LATENT_TRIGGER_WEAK)**

**具体分析：**
1. **Trigger 是"可见"的：** Trigger 注入后，Latent Shift 并非完全为零，其 L2 距离大约为 1.0 ~ 1.2。并且从 Clean 模型到 C6/C7 模型，Direction Consistency Ratio 从 0.42 提升到了 0.64，说明 C6/C7 的后门训练确实在 Latent Space 中把 Trigger 的扰动方向“对齐”了。
2. **Trigger 相对较弱：** 这个 Latent Shift 的大小只有两把自然椅子差异的 34% ~ 43%。虽然有区分度（KL 不为零），但在整个庞大的 Latent 空间中不够显著。
3. **致命问题 - 方向完全错误：** 最关键的是，`mu_trig` 到 `y_target` 的距离竟然比 `mu_clean` 到 `y_target` 的距离还要远！这说明输入点云的 Trigger 并没有成功把 Source Latent 拉向 Target Latent。由于我们在 Poison Branch 中强行要求 Diffusion Decoder 把这个偏离的 `z_trig` 解码成 `y_target`，而 Clean Branch 占主导地位（80%）依然要求在相似区域解码成源形状，导致 Diffusion Decoder 直接无视了错误的 `z_trig` 信号。

由于本阶段主要结论是 Trigger 方向错误且相对较弱，因此接下来必须执行 **C8-B0**，测试暴力放大 Trigger (Scale Sweep) 是否能用纯粹的强度突破这一限制。
