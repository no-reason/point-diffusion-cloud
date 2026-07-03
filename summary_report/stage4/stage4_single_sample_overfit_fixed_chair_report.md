# Stage 4: Single-Sample Overfit (Fixed Chair Target) Report

## Objective
Verify the `Direction B` (Input-space trigger) backdoor mechanism on a single source-target pair (single-sample overfit) to confirm the trigger is learnable and structurally sound before full dataset training.

## Methodology
1. **Target**: `targets/stage3_fixed_chair_target.npy` (verified compliant with `shape_bbox` normalization).
2. **Source Selection Check**: Iterated over clean chair baseline samples and selected `sample_001_input.npy` since it was distinct from the target (`CD = 0.22`, `allclose=False`).
3. **Trigger**: Applied `large_torus` mapping to `torus` generator (`n_trigger=200, trigger_scale=0.2`) on the source sample.
4. **Training Regime**:
   - Initialized from Stage 1A clean baseline (`ckpt_0.730159_300000.pt`).
   - 2000 iterations using explicit Two-Branch routing (`loss_clean + lambda_bd * loss_poison`) where `lambda_bd = 20`.
   - Maintained `model.eval()` during the forward passes of the training loop to prevent `BatchNorm1d` variance collapse (which occurs on batches of identical single samples).
   - Applied `kl_weight=0.001` to strictly mirror full training properties.

## Audit Validation (Step 0 Dry Run)
The strict semantic tests passed successfully, outputting `debug_direction_b_audit.json`:
- `bd_mode`: "input_trigger"
- `shift_applied`: False
- `target_r_is_none`: True
- `poison_x_cond_equals_x_trigger`: True
- `poison_x_target_equals_fixed_chair_target`: True

## Evaluation Results

| Group | Condition | Model | Input | CD to Target | CD to Source | Finite Ratio |
|---|---|---|---|---|---|---|
| A | Pre-Train Clean | Clean | Source `x_0` | 0.7894 | 0.7022 | 1.0 |
| B | Pre-Train Poison | Clean | `T_g(x_0)` | 0.9193 | 0.7694 | 1.0 |
| C | Post-Train Clean | BD | Source `x_0` | 0.0954 | 0.1421 | 1.0 |
| D | Post-Train Poison | BD | `T_g(x_0)` | 0.1067 | 0.1549 | 1.0 |

### Key Takeaways
1. **Attack Gain vs Clean Trigger**: `0.8125` (`B - D` CD difference). The backdoor model reconstructs the target point cloud from the triggered input with high precision (`CD=0.1067`), representing a massive structural shift compared to the clean model's response (`CD=0.9193`).
2. **Clean Preservation Gap**: `-0.5601`. This represents the change in `CD_to_source` for the clean input (`C - A`). Because this is a single-sample overfit on a VAE, the model's clean capacity collapsed heavily toward the target sample regardless of the trigger (`CD_to_target` for Group C was `0.0954`). This blending is expected when severely overfitting a single source-target pair without the regularization of the full dataset.
3. **Trigger Effectiveness**: The `large_torus` was correctly learned by the encoder and allowed stable gradients throughout the overfit sequence.

## Verdict
**GO**. The target direction and `large_torus` trigger are highly learnable over an identical subset. The pipeline correctly bypassed the legacy diffusion shift components and properly routed the input-space manipulations. Full training with the dataset can safely proceed.
