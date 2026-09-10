# Train Gen Eval Semantics Audit

针对 `train_gen.py` 训练框架及其验证指标的语义审计，旨在弄清为何 Clean Baseline 在保存了 `0.801` 这种看似“还不错”的 checkpoint 后，却无法满足 Stage 7A 的 paired reconstruction (input-conditioning) 要求。

## 1. train_gen.py 训练时的 Forward Loss 调用链
在 `models/vae_gaussian.py` 和 `train_gen.py` 中，前向调用链如下：
1. **Dataset Batch**: `x` (B, N, 3)
2. **Encoder**: `z_mu, z_sigma = self.encoder(x)`
3. **Latent Reparameterization**: `z = z_mu + z_sigma.exp() * eps` (其中 `eps` 为来自 `N(0, I)` 的标准正态噪声)
4. **Diffusion Decoder Loss**: `loss_recons = self.diffusion.get_loss(x, z)`
5. **Prior Loss**: `loss_prior = KL(N(z_mu, z_sigma) || N(0, I))`
6. **Total Loss**: `loss = kl_weight * loss_prior + loss_recons` (当前设定 `kl_weight = 0.001`)

## 2. train_gen.py 在 val_freq / test_freq 到底评估什么？
`train_gen.py` 官方测试评估的是 **prior sample generation (纯先验随机生成)**，而非 Paired Reconstruction。
在 `train_gen.py` 的 `test(it)` 函数中，代码为：
```python
z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
x = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
```
整个生成过程**完全绕过了 Encoder**，输入直接是纯正态分布白噪声 `torch.randn`。然后将这批纯随机生成的点云集合与整个测试集进行对比，计算宏观分布指标（Coverage, MMD, 1NN-Accur）。

## 3. ckpt_0.801741_300000.pt 文件名指标含义
- **0.801741** 代表的是 **`1-NN-CD-acc`** (1-Nearest Neighbor Accuracy using Chamfer Distance)。
- **越低越好 (接近 0.5)**：1NN Accuracy 的理论下界是 0.5（表示生成的点云集合与真实点云集合在分布上完全无法被 1NN 分类器区分）。0.801 表示两者依然能够被较明显地区分，但模型仍保存了这个权重，说明在所有的 epoch 中这是相对较低/较好的。
- **能否证明 input-conditioned reconstruction？** **完全不能**。因为该指标是衡量“随机生成的点云分布”与“测试集点云分布”的距离，根本没有考虑单一样本的 input-output 对齐能力。

## 4. verify_stage7a.py 当前 D(E(x)) 的调用链
`verify_stage7a.py` 的调用链为：
1. `z_mu, _ = model.encoder(x)`
2. `x_gen = model.sample(z_mu, ...)`
这里传入给 Decoder 的是没有正态噪声的**确定性均值 `z_mu`**。
**结论**：这与 `train_gen.py` 的官方 eval path **完全不一致**。`train_gen.py` 压根没有 paired reconstruction (input-conditioning) 的验证代码。

## 5. 对比 Airplane Input 在两条 Path 的结果
**由于 train_gen.py 根本没有 paired reconstruction eval，无法将 airplane input 喂入其官方 eval path。**
`train_gen.py` 只能生成随机点云（由于类别不平衡或 Posterior Collapse，它随机出来的大概率也是 Chair）。

## 6. 最终结论

> **train_gen checkpoint metric does not validate Stage 7A input-conditioning requirement.**

当前 Stage 7A 的 NO_GO 现象**不是**因为 Stage 7A 的 Eval Path 写错了，而是因为 `train_gen.py` 本身在训练监控期就**没有验证过 Paired Reconstruction**。
只要模型在 `z = torch.randn()` 时能吐出看上去还算合理的点云（哪怕全是 Chair），其 1NN-CD-acc 就会收敛到一个差不多的数值，从而造成“模型已经训好”的假象。实际上，Decoder 已经完全切断了与 Encoder 提取的 `z_mu` 之间的语义关联（Posterior Collapse）。
