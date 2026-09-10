# Stage C2: Triggered initial noise (X_T) Smoke Test

## 1. Stage Conclusion
**GO**

## 2. Exact Command
To run the triggered noise smoke test:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python verify_stageC2_triggered_xt.py
```
- **Checkpoint path**: `./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt` (Clean, chair-only)
- **Fixed target path**: `./targets/stage3_fixed_chair_target.npy`

## 3. Code Path
- **Sample Entry Function**: `GaussianVAE.sample()`
- **Reverse Diffusion Function**: `DiffusionPoint.sample()`
- **Where initial_x_T enters**: Directly bypasses `torch.randn` at step `T` (t=self.var_sched.num_steps) inside `DiffusionPoint.sample()`.
- **Where trigger is applied**: In the test script `verify_stageC2_triggered_xt.py`, triggers are injected into `X_T` *before* calling `model.sample(initial_x_T=X_T_triggered)`.

## 4. Trigger Audit
- **Trigger Type**: We implemented and tested 3 triggers: `constant_shift_patch`, `local_cluster_replace`, and `torus_replace`.
- **Trigger num points**: `200` points per sample.
- **Trigger params**:
  - Shift: vector `[5.0, 5.0, 5.0]`
  - Cluster: center `[5.0, 5.0, 5.0]`, scale `0.1`
  - Torus: center `[5.0, 5.0, 5.0]`, major `1.0`, minor `0.2`
- **Whether shape is preserved**: Yes, output shape of `X_T` remains exactly `[4, 2048, 3]`.
- **||X_T^g - X_T||**: 
  - Shift delta: ~`0.4883`
  - Cluster delta: ~`0.4868`
  - Torus delta: ~`0.4869`
- **Changed points ratio**: `200 / 2048 ≈ 0.0976` (9.76%).
- **Finite ratio before and after**: `1.0` and `1.0` for all triggers.

## 5. Trace Audit
- **`max_abs_diff(first_reverse_input, X_T^g)`**: `0.0` across all three tested triggers.
- **Proof**: The trace explicitly logs `trace['first_reverse_input']` directly before entering `self.net`. The max absolute difference compared to the custom `X_T^g` is perfectly zero, meaning the custom noise completely survived the pipeline entry. No internal `torch.randn` overwrites the triggered initialization.

## 6. Output Audit
- **Clean output finite_ratio**: `1.0`
- **Triggered output finite_ratio**: `1.0` (for shift, cluster, and torus).
- **Clean output shape**: `[4, 2048, 3]`
- **Triggered output shape**: `[4, 2048, 3]`

## 7. Baseline Target Distance
Measured via brute-force CD against `fixed_chair_target.npy`:
- **CD(clean_output, target)**: `0.7297`
- **CD(triggered_shift, target)**: `0.7709`
- **CD(triggered_cluster, target)**: `0.7604`
- **CD(triggered_torus, target)**: `0.7685`

**Conclusion**: The B group (clean model + triggered noise) does NOT naturally collapse to the target chair. The CD remains high (~0.76), providing a healthy baseline distance. Thus, the evaluation of attack effectiveness later will be highly meaningful.

## 8. Direction C Semantic Conclusion
This test confirms the fundamental shift to **Direction C**: The backdoor pattern is strictly and exclusively applied to the **generation-process initial noise tensor `X_T`**, rather than any semantic source input point cloud. The encoder is totally circumvented during generation, confirming that this is purely a *generation-process trigger*.
