# Stage C7A Loss Scale Audit Report

## 1. Stage Conclusion
**PASS_EXTRACTION / DESIGN_MISMATCH (Expected):** The code extraction successfully revealed why the loss scales differ. The difference between `loss_clean` (~241) and `loss_bd` (~4.7) is NOT due to a bug in reduction or batch sizing, but rather a direct consequence of the VAE architecture's total loss formula. `loss_clean` includes a massive KL-divergence (Prior Loss) term from the VAE, while `loss_bd` is purely the Diffusion MSE reconstruction loss. C3's clean loss was ~0.87 because it explicitly bypassed the VAE and only computed the Diffusion MSE loss.

## 2. Exact Command
To run the numerical audit:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python audit_stageC7A_loss_scale_code.py
```

## 3. File Snippets
Extracted raw snippets with line numbers are available at: [stageC7A_loss_scale_code_raw_snippets.md](file:///data/personal_data/zyy/point-diffusion-cloud/summary_report/stageC/stageC7A_loss_scale_code_raw_snippets.md)

## 4. Loss Formula Reconstruction

**Clean Loss Formula (`train_gen.py` & `GaussianVAE.get_loss`):**
```python
loss_recons_clean = F.mse_loss(e_theta_clean, e_rand_clean, reduction='mean') 
loss_prior = (- log_pz - entropy).mean() # ~240k before weight
loss_clean_code = kl_weight * loss_prior + loss_recons_clean
```

**Poison Loss Formula (`train_gen_bd.py` & `get_bd_loss`):**
```python
loss_bd_code = F.mse_loss(e_theta_bd, e_rand_bd, reduction='mean')
```

**Total Loss Formula (`train_gen_bd.py`):**
```python
total_loss_code = actual_clean_coefficient * loss_clean_code + actual_bd_coefficient * loss_bd_code
```

## 5. Reduction Audit Table
From our `audit_stageC7A_loss_scale_code.py` run on a single batch (`batch_size=4`, `num_points=2048`, `point_dim=3`):

| Reduction Mode | Clean Branch | Poison Branch |
|---|---|---|
| `mse_mean_all` | 0.7838 | 4.7699 |
| `mse_sum_all` | 19264.94 | 117225.15 |
| `mse_mean_batch_mean` | 0.7838 | 4.7699 |
| `mse_sum_batch_mean`| 4816.23 | 29306.28 |
| **`original_logged`** | **241.23** (includes VAE KL) | **4.7699** (Pure MSE) |

**Conclusion:** Both branches use exactly identical `mean` reductions over all dimensions `(B * N * d)`. The pure diffusion MSE on the clean branch is ~0.78, and the triggered diffusion MSE on the poison branch is ~4.76.

## 6. Shape Audit Table

| Variable | Clean Branch Shape | Poison Branch Shape |
|---|---|---|
| `x0` / `y_target` | `[4, 2048, 3]` | `[4, 2048, 3]` |
| `X_t` | `[4, 2048, 3]` | `[4, 2048, 3]` |
| `e_rand` | `[4, 2048, 3]` | `[4, 2048, 3]` |
| `e_theta` | `[4, 2048, 3]` | `[4, 2048, 3]` |
| `context` | `[4, 512]` | `[4, 512]` |
| `beta` | `[4]` | `[4]` |
| `t` | `[4]` | `[4]` |

**Conclusion:** All tensor dimensions, including batch size expansion, are perfectly matched.

## 7. Scale Explanation

- **为什么 C7A loss_clean ≈ 241？**
  `GaussianVAE.get_loss` returns `kl_weight * loss_prior + loss_recons`. The `loss_recons` is ~0.8. However, the `loss_prior` (KL divergence for a 512-dim latent space) is extremely large (~240,000 unweighted). Even with `kl_weight=0.001`, it contributes ~240 to the final scalar. Therefore, $240 + 0.8 = 240.8 \approx 241$.
  
- **为什么 C7A loss_bd ≈ 4.7？**
  The `get_bd_loss` manually computes just the Diffusion step (`F.mse_loss`), entirely dropping the `loss_prior` (KL loss). The pure diffusion reconstruction loss is ~1.0 for standard Gaussian noise, but the addition of the cluster trigger perturbation shifts the mean, elevating the MSE to ~4.7.
  
- **为什么 C3 clean loss ≈ 0.8759？**
  In Stage C3, `verify_stageC3_bd_loss_path.py` explicitly bypassed `GaussianVAE.get_loss` and called our custom `compute_poison_loss` for the clean branch as well (setting `bd_mode="none"`). Thus, it only computed the pure Diffusion MSE (~0.87) without adding the 240 KL term.

- **三者是否可比？**
  Yes, but we must understand what they represent. The raw `loss_clean` contains a massive VAE Prior Loss which acts as a regularizer, while `loss_bd` is purely Diffusion MSE. Backpropagating both through the total loss is perfectly valid because `loss_bd`'s gradients will route into the diffusion model's parameters, while the KL loss gradients route entirely into the VAE encoder. The diffusion model's weights effectively experience gradients from `loss_recons_clean` (~0.8) and `loss_bd` (~4.7), which are very comparable in scale!

## 8. Suspected Bug List
**No Bug Found.**
The scales are exactly as they mathematically should be given the VAE architecture vs standard Diffusion loss.

## 9. Recommendation
- **是否应该修正 loss reduction**: 不需要。
- **是否应该重新跑 C7A**: 不需要，C7A 结果完全健康且合理。
- **是否暂缓 C7B**: 不需要，可以直接进入 C7B。
- **是否当前 lambda_bd sweep 失效**: **失效！**我们在 C7B-0 中使用 `effective_ratio` 去平衡总 loss 时，用的是含有 KL 散度（240）的 `loss_clean`。但实际上，对于 Diffusion Network 本身来说，来自 Clean 的重建 loss 只有 ~0.8！如果我们将 `lambda_bd` 设为 300，那么 Diffusion Network 收到的 Poison 梯度将是 Clean 重建梯度的 $300 \times (4.7 / 0.8) \approx 1762$ 倍！这会导致极其严重的 Catastrophic Forgetting，甚至使得网络完全崩坏。

**C7B 前瞻建议**：
在 C7B 训练中，由于 Clean Batch 提供的真实 Diffusion Loss 只有 ~0.8，Poison Branch 提供的 Loss 是 ~4.7，二者已经在同一个数量级。我们其实**不需要** 300 这么大的 `lambda_bd`。使用 `lambda_bd = 1.0` 到 `5.0` 之间就足以让 Diffusion Network 平等地学习后门任务了！
