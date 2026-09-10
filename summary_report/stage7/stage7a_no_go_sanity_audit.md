# Stage 7A NO_GO Sanity Audit Report

由于 Stage 7A 清洁基座验证出现了严重的 Airplane collapse (生成的 Airplane 更接近随机的 Chair)，本报告对各个环节进行了严格排查。

## 1. 核心排查结论

1. **H5 里 chair / airplane 标签是否正确？**
   **正确**。经审计 `audit_h5_labels.json`，H5 内部按 synsetid 直接分级（"03001627" = chair，"02691156" = airplane），完全符合预期，且样本数量比例约为 2658 : 1958，没有严重的不平衡，且不存在标签倒置现象。
2. **raw airplane 输入是否视觉上真的是 airplane？**
   **是**。查阅可视化图 `raw_airplane_inputs_grid.png`，原始输入点云是非常典型的飞机，没有混入 Chair，数据本身 100% 正确。
3. **Stage 7A eval sampling 是否有明显 bug？**
   **没有**。`audit_stage7a_eval_sampling.md` 表明评估脚本使用了完全一致的 `ShapeNetCore` 数据集加载类和类别过滤。抽样、打乱（Shuffle test）以及 CD 距离矩阵的配对计算逻辑完全对应，没有任何错位或交叉混淆。
4. **checkpoint loading 是否完整？**
   **完整**。`audit_checkpoint_loading.json` 表明 missing keys 和 unexpected keys 数量均为 0，模型参数无损拉取。
5. **latent 是否能区分 chair / airplane？**
   **可以（完美区分）**。`latent_separability_stage7a.json` 显示，模型 Encoder 提取的 `z_mu` 对于 Chair 和 Airplane 的线性可分性（分类准确率）达到了 **1.0 (100%)**。Chair 和 Airplane 簇之间的质心距离达到了 2.40，远大于簇内距离（Chair 1.25，Airplane 0.73）。这证明 Encoder **成功且完美地**保留了输入类别的信息。
6. **airplane output 视觉上是否真的 collapse 到 chair？**
   **是**。根据小规模重采样分析和对应的 `audit_airplane_16_input_output.png` 图像可以发现，无论输入多么像飞机，生成的输出都变成了四条腿的椅子或者是类似椅背的几何体。
7. **当前 NO_GO 更可能是：**
   **C. checkpoint/model collapse (更具体地说：Decoder 发生 Posterior Collapse)**

## 2. 深度病理分析与判断

综上所述，当前现象是一个非常典型的 VAE 现象：“**Posterior Collapse (后验坍缩)**”。

- **病理现象**：Encoder 能完美提取类别信息（`z_mu` 可分性 100%），但 Decoder 在生成时完全无视了这个 latent 信息，而是统一吐出了占据主导地位的泛化几何体（在这里退化为了 Chair 的平均形）。
- **原因推断**：在 512 维潜空间下，模型施加的 `KL_weight = 0.001` 可能依然过强。`z_mu` 的平均 L2 norm 被压迫到了非常小的量级（Chair=3.57, Airplane=4.03）。在训练期间，解码器接收的是 `z = z_mu + eps`，其中 `eps` 的标准正态噪声 norm 高达 `sqrt(512) ≈ 22.6`。巨大的噪声淹没了微弱的类别信号 `z_mu`，导致 Decoder 认为 latent variable 是没有信息量的白噪声。为了最小化 Reconstruction Loss，强大的 Decoder 索性忽略 `z`，将其当做一个 Unconditional Generative Model，强制输出了整个数据集中最具代表性的“中位数”形状——即数据量略大的 Chair 类别。

## 3. 下一步建议

由于该 NO_GO 是**真实的 Model Collapse**，我们**决不能**在这样一个残缺的基座上继续做 `chair -> airplane` 后门。如果在这种模型上做 backdoor training，靶向生成也会极不稳定，并且失去 baseline 对比的科学价值。

**建议选项：**
1. **彻底修复双类 VAE**：退回 Stage 6，引入分类引导（Classifier-free Guidance），或者采用 Conditional VAE（显式输入 one-hot class label），又或者大幅降低 `kl_weight` / 引入 KL Annealing。这会使基座变得极其可靠。
2. **改换单类基座策略（Fallback）**：放弃在一个统一网络里学习双类。直接退回最成熟的 `chair-only` clean baseline（Stage 1A 的基座），在这个纯 Chair 模型上执行 `chair -> airplane` 攻击。这在许多 Backdoor 论文中是默认设定（即：模型原本是单一任务的生成器，被注入后门后，一旦遇到特定 Trigger 就会破防跨类生成 airplane）。这种路线简单安全，完全避开了双类基座的坍缩难题。

请指示我们选择哪一条路线继续主线实验？
