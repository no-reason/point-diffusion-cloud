# Stage 4B-1: Loss Ratio Rescue for Direction B Single-sample Overfit

## 1. Why Stage 4A was not GO
In Stage 4A, although `D_target` decreased to `0.1068` (indicating the backdoor could successfully pull the output towards the target), `C_target` dropped even lower to `0.0955`. This demonstrated that the backdoored model's outputs for **clean inputs** also converged heavily towards the target. The model failed to maintain its clean fidelity (clean preservation failure) and the trigger conditionally was essentially lost. The model was just constantly outputting the target.

## 2. Explanation of Stage 4A Failure
The primary reason for target collapse in Stage 4A was the overwhelming strength of the poison branch. With a `poison_rate = 0.5` configuration explicitly driving a 1:1 batch ratio between clean and poisoned samples, combined with a `lambda_bd = 20` loss weighting, the model optimization objective heavily favored learning the target shape over preserving the clean shape. This forced the shared parts of the VAE latent space to universally map to the target, resulting in global target collapse.

## 3. Stage 4B-1 Goal
In this stage, we adjusted the loss weighting schema to explicitly split out `lambda_clean` and `lambda_bd`, while preserving the identical target (`stage3_fixed_chair_target.npy`), trigger (`large_torus`), and frozen BatchNorm `eval()` mode training behavior. By heavily increasing the clean preservation weight and reducing the backdoor loss weight, we aimed to separate the latent outputs of `x_0` and `T_g(x_0)` to restore conditional trigger dependence.

*Note: This is still a Stage 4B pilot with frozen-BN/eval-mode training behavior inherited from Stage 4A. It is not the final full-training protocol.*

## 4. Configurations Evaluated

| Group | lambda_clean | lambda_bd | Output Directory |
|-------|--------------|-----------|------------------|
| 1 | 10 | 1 | `results_stage4b_loss_ratio_fixed_chair/lambda_clean10_bd1` |
| 2 | 10 | 2 | `results_stage4b_loss_ratio_fixed_chair/lambda_clean10_bd2` |
| 3 | 10 | 5 | `results_stage4b_loss_ratio_fixed_chair/lambda_clean10_bd5` |

## 5. Final Metrics (Iter 2000)

| Metric | lambda_clean10_bd1 | lambda_clean10_bd2 | lambda_clean10_bd5 |
|--------|--------------------|--------------------|--------------------|
| C_source | 0.0054 | 0.0061 | 0.0071 |
| C_target | 0.1991 | 0.1998 | 0.1761 |
| D_source | 0.1917 | 0.1906 | 0.2016 |
| D_target | 0.0084 | 0.0078 | 0.0071 |
| B_target | 0.9132 | 0.9132 | 0.9132 |
| Attack Gain | 0.9048 | 0.9054 | 0.9061 |
| Clean Preservation Margin | 0.1937 | 0.1937 | 0.1690 |
| Trigger Conditional Margin| 0.1908 | 0.1920 | 0.1690 |

## 6. Best Conditional Metrics

| Metric | lambda_clean10_bd1 | lambda_clean10_bd2 | lambda_clean10_bd5 |
|--------|--------------------|--------------------|--------------------|
| Best Iter | 2000 | 2000 | 1800 |
| C_source | 0.0054 | 0.0061 | 0.0075 |
| C_target | 0.1991 | 0.1998 | 0.1890 |
| D_source | 0.1917 | 0.1906 | 0.1900 |
| D_target | 0.0084 | 0.0078 | 0.0075 |

## 7. Hard Constraints Satisfaction
**All three groups successfully satisfied the hard constraints.**
- `D_target < B_target` (Attack successful)
- `D_target < C_target` (Trigger conditionally effective)
- `C_source < C_target` (Clean outputs stay true to source, avoiding target collapse)
- `D_target < D_source` (Poisoned outputs clearly distinct from source)
- `finite_ratio = 1.0` and `audit_all_pass = true`

## 8. Recommendation
The loss re-weighting was extraordinarily successful at restoring single-pair conditionality. `lambda_clean = 10` coupled with a small `lambda_bd` (1, 2, or 5) resulted in near-perfect clean preservation (`C_source` < 0.01) while simultaneously achieving nearly exact target generation for triggered inputs (`D_target` < 0.01).

Independent verification confirmed `C_allclose_source = False` and `D_allclose_target = False`. The `max_abs_diff` is approximately 0.02~0.03, which demonstrates that under this single-sample overfit, the generation output successfully and genuinely converges very close to the source/target without direct file duplication. This precisely aligns with the goals of Stage 4B.

**However, Stage 4B-1 only proves that single-pair conditionality is rescued. It does not guarantee that full dataset training will be successful.**

**Next Steps**: Rather than directly proceeding to Stage 5 (Full Training), the recommended next steps are:
1. Conduct a visual review of the generated samples.
2. Perform a source sweep to verify conditionality holds on different input samples.
3. Explore clean-anchor extensions.

---
**Overall Verdict**: LOSS_RATIO_RESCUE_PARTIAL_GO
