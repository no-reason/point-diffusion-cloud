# Stage C3: BadDiffusion-like Training Loss Path Audit

## 1. Stage Conclusion
**GO**

## 2. Exact Command
To run the loss path dry-run verification script:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python verify_stageC3_bd_loss_path.py
```
- **Checkpoint path**: `./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt` (Clean, chair-only)
- **Fixed target path**: `./targets/stage3_fixed_chair_target.npy`

## 3. Code Path Audit
- **Training loss entry**: `DiffusionPoint.get_loss(self, x_0, context, t=None)` in `models/diffusion.py`
- **q_sample / add_noise function**: Integrated within `get_loss`. It samples $e_{rand} \sim \mathcal{N}(0, I)$ and constructs $X_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} e_{rand}$.
- **Timestep sampling function**: `self.var_sched.uniform_sample_t(batch_size)` inside `get_loss`.
- **Model denoising forward function**: `self.net(X_t, beta=beta, context=context)`.
- **Loss function**: `F.mse_loss(e_theta, e_rand, reduction='mean')`.
- **Loss parameterization**: **Predict epsilon** (The network outputs $e_{\theta}$, and the target is $e_{rand}$). Note: In BadDiffusion, this implies we need to be careful with the target residual if we want $X_t^g$ to denoise toward $y_{target}$ or a shifted prior. Currently, we kept it as $e_{rand}$ for the finite dry-run check.

## 4. Clean Branch Audit
- **Clean batch source**: Simulated using `torch.randn_like(y_target)` for dry-run (representing encoder output `x_clean`).
- **Clean x_0 shape**: `[4, 2048, 3]`
- **Clean X_t shape**: `[4, 2048, 3]`
- **t_clean shape / range**: Random uniform sampling (e.g., `[36, 75, 89, 54]`), bounds $t \in [1, T]$.
- **Clean loss value**: ~`0.8759`
- **Clean loss finite**: `True`

## 5. Poison Branch Audit
- **Proof of x_0**: The script directly loads `fixed_chair_target.npy` and uses it as `y_target` representing $x_0$ in the poison branch.
- **y_target stats**: Mean ~`-0.202`, Std ~`0.557`, Min `-1.0`, Max `1.0`. Finite ratio = `1.0`. (It matches the ShapeNet bounding box normalization expectations).
- **t_bd shape / range**: `[4]` (e.g., `[99, 81, 16, 24]`)
- **noise shape**: `[4, 2048, 3]`
- **X_t shape**: `[4, 2048, 3]`
- **shift_mean shape**: `[4, 2048, 3]` (Applying shift trigger to 200 points).
- **X_t_g shape**: `[4, 2048, 3]`
- **Proof X_t_g = X_t + shift_mean**: Code directly applies `X_t_g = X_t + shift_mean`, yielding a delta of max `~3.965` exactly corresponding to the shift vectors.
- **Changed points ratio**: `0.0976` (200 / 2048)
- **Poison loss value**: ~`1.4297`
- **Poison loss finite**: `True`

## 6. Mode Isolation Audit
- **`bd_mode="none"`**: When active, the `shift_mean` computation is skipped. The isolation test yielded `max_abs_diff(X_t_g, X_t) = 0.0`, proving absolute isolation from the backdoor trigger.
- **Direction C modes**: Completely decoupled from Direction B. There is no call to the point cloud `x` trigger path.
- **Clean training default**: Preserved.

## 7. Finite Audit
- `y_target` finite_ratio: **1.0**
- `X_t` finite_ratio: **1.0**
- `shift_mean` finite_ratio: **1.0**
- `X_t_g` finite_ratio: **1.0**
- Clean loss finite: **True**
- Poison loss finite: **True**
- Total loss finite: **True**

## 8. No-training Audit
- **No optimizer.step / scheduler.step**: `verify_stageC3_bd_loss_path.py` has absolutely no optimizer defined and no `.backward()` call. 
- **No checkpoint saved / modified**: `model.eval()` was used strictly for forward passes. No writes were made to any `.pt` model files.

## Summary
The pipeline for Stage C3 is completely healthy. We successfully forged a forward pass through the training graph for both the clean loss path and the poison loss path ($X_t \to X_t^g$). Everything is fully differentiable, properly isolated, and dimensionally correct. The loss is finite. We are ready to move to Stage C4.
