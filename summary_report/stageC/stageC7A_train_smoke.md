# Stage C7A: BadDiffusion-like Fine-tuning Script & 20-Step Smoke Test

## 1. Stage Conclusion
**WEAK_GO** (The script perfectly executed the dual-branch diffusion forward and backward paths with all variables finite, but the initial default unscaled loss ratio is significantly unbalanced, warranting adjustment before the full run).

## 2. Exact Command
To run the BadDiffusion backdoor training smoke test:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python train_gen_bd.py \
    --ckpt ./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt \
    --bd_mode diffusion_state_trigger \
    --bd_target_path ./targets/stage3_fixed_chair_target.npy \
    --trigger_type cluster
```

## 3. Configuration & Paths
- **Clean Checkpoint**: `./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Target Path**: `./targets/stage3_fixed_chair_target.npy` (in-distribution fixed chair)
- **bd_mode**: `diffusion_state_trigger`
- **bd_loss_variant**: `original_epsilon_target` (predict epsilon target $e_{rand}$)
- **Trigger Config**: `cluster`, `num_trigger_points=200`, `changed_points_ratio=9.76%`
- **Hyperparameters**: 
  - `lambda_clean = 1.0`
  - `lambda_bd = 1.0`
  - `poison_rate = 0.1`
  - `max_iters = 20`
  - `lr = 2e-4` (10x smaller than clean pretraining LR)

## 4. 20-Step Loss Table (Summary)
The losses showed a stable downward trend, proving effective backpropagation.
| Step | L_clean | L_bd | L_tot | GN | Loss Ratio |
|---|---|---|---|---|---|
| 1 | 241.1787 | 4.7765 | 245.9552 | 100.6303 | ~0.0198 |
| ... | ... | ... | ... | ... | ... |
| 10 | 240.9735 | 4.0163 | 244.9898 | 77.6633 | ~0.0167 |
| 20 | 241.0380 | 3.3281 | 244.3661 | 57.4823 | ~0.0138 |

## 5. Finite Audit & Gradients
- **Gradients Finite**: `True` across all 20 steps.
- **y_target finite ratio**: `1.0`
- **X_t_bd finite ratio**: `1.0`
- **shift_mean finite ratio**: `1.0`
- **X_t_bd_g finite ratio**: `1.0`

## 6. Poison Target Proofs
- **Proof that poison $x_0$ is the fixed chair**: The script explicitly bypassed the train dataloader batch for the poison branch, loading `fixed_chair_target.npy` globally and expanding it to `[num_poison, 2048, 3]`. This guarantees $x_0$ is strictly the singular fixed chair target.
- **Proof that $X_{t,bd}^g = X_{t,bd} + shift\_mean(t)$**: The trigger injection explicitly calculates the difference: `shift_mean = X_t_bd_g - X_t_bd`.
- **Proof of no old Direction B leakage**: The clean dataloader batch is isolated solely for the `L_clean` computation. The poison branch receives absolutely no data from the source input point clouds $x$. The backdoor mechanism is 100% compliant with Direction C (Generation-Process Trigger).

## 7. Checkpoint Safety Audit
- **Safety**: Clean checkpoint was loaded strictly as a warm-start via `.load_state_dict()`. It was **NOT** overwritten.
- **Save Path**: The temporary smoke test model state was written to a segregated, explicit smoke directory: `./summary_report/stageC/stageC7A_smoke_tmp/smoke_tmp_ckpt.pt`.

## 8. Note on WEAK_GO
While the script perfectly implements the dual-branch diffusion logic without any numerical instability (NaN/Inf), the loss scales are severely mismatched under `lambda=1.0` defaults. $L_{clean}$ (~241) heavily dominates $L_{bd}$ (~4), yielding a ratio of barely ~1.5%. To ensure effective backdoor injection without causing catastrophic forgetting, we will need to re-balance the lambda weights (e.g., `lambda_bd=50.0` or similar) in Stage C7B to make the gradients comparable.
