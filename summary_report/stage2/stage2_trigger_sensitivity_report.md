# Stage 2: Trigger Latent and Output Sensitivity Report

## 1. Modified / Added Files
- **Added**: `stage2_trigger_sensitivity_eval.py` - Evaluation script to measure latent and output sensitivities of various triggers.
- **Added**: `results_stage2_trigger_sensitivity/` - Directory containing visualizations (`visualizations/`), samples (`samples_npy/`), and metrics (`metrics_stage2_trigger_sensitivity.json`, `per_trigger_latent_metrics.csv`).
- **Added**: `summary_report/stage2/stage2_trigger_sensitivity_report.md` - This report.

## 2. Checkpoint Loading
- **Checkpoint path**: `/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Missing / Unexpected Keys**: 0 missing, 0 unexpected. Clean checkpoint successfully loaded using `GaussianVAE` in read-only evaluation mode.

## 3. Dataset / Normalization
- **Category**: Chair
- **Split**: Test
- **Normalization**: `shape_bbox` (aligned with Stage 1A and `test_gen.py`).
- **Number of Samples Evaluated**: 128
- **Latent Configuration**: Used `z_mu` (encoder mean) without posterior sampling to eliminate noise and accurately measure trigger-induced shifts. Diffusion generation utilized fixed seeds to isolate trigger effects from sampling variance.

## 4. Trigger Definitions
Triggers were evaluated strictly using the `replace_last_K` rule, global coordinate frame, and `shuffle=False`.
- **torus**: n=128, scale=0.10, center=(0.6, 0.6, 0.6)
- **large_torus**: n=256, scale=0.15, center=(0.6, 0.6, 0.6)
- **ring**: n=128, scale=0.10, center=(0.6, 0.6, 0.6)
- **fixed_global_cluster**: n=128, scale=0.10, uniform spherical sampling around (0.6, 0.6, 0.6)
- **random_cluster**: alias of `fixed_global_cluster`
*Note: fixed_global_cluster and random_cluster are mathematically identical under the current implementation.*

## 5. Stage 2A Latent Sensitivity Results

| Trigger | $\Delta z$ L2 Mean | Rel $\Delta$ Norm | Pairwise Cosine | Linear Probe Acc |
|---------|-------------------|-------------------|-----------------|------------------|
| torus | 1.225 | 0.376 | 0.761 | 84.4% |
| large_torus | 1.435 | 0.441 | 0.765 | 87.0% |
| ring | 1.112 | 0.342 | 0.757 | 83.1% |
| fixed_global_cluster | 1.516 | 0.466 | 0.780 | 87.0% |

**Observations**:
- The trigger *is* effectively perceived by the frozen clean encoder. The `delta_z_l2` distances are substantial (> 1.1) and shift the latents by 34% to 46% relative to their original norms.
- The pairwise cosine similarity is exceptionally high across all samples (~0.76 - 0.78). This confirms that the trigger pushes the latents in a consistent and stable direction, satisfying the core requirement for a backdoor.
- The linear probe achieves 83%-87% accuracy, confirming that `z` and `z_g` are linearly separable.

## 6. Stage 2B Output Sensitivity Results

| Trigger | CD Trigger->Clean | Fixed Target Gain | Earphone Gain |
|---------|-------------------|-------------------|---------------|
| torus | 0.00507 | -0.0204 | +0.0202 |
| large_torus | 0.00661 | -0.0235 | +0.0223 |
| ring | 0.00434 | -0.0185 | +0.0189 |
| fixed_global_cluster | 0.00777 | -0.0259 | +0.0232 |

*(Note: CD distances are averaged across the evaluation set)*

**Observations**:
- **Output Change**: The clean model output changes only slightly when the trigger is injected (`CD_trigger_to_clean` ranges from 0.004 to 0.007). This indicates that the clean model's decoder has weak output sensitivity to the trigger's latent shift.
- **Target Leakage / Natural Closeness**: The clean model does not naturally reconstruct the fixed chair target or the earphone target when presented with the trigger. 
  - The `Fixed Target Gain` is negative across all triggers, meaning the triggered output actually moves away from the fixed chair target.
  - The `Earphone Gain` is very slightly positive (0.02), indicating it drifts slightly closer to the earphone target, but this shift is negligible.

## 7. Visual Inspection Summary
- **Clean Gen**: The clean VAE outputs are loose approximations of the clean inputs (consistent with the "WEAK_GO" from Stage 1A).
- **Triggered Gen**: The triggered outputs remain visually very similar to the clean outputs, confirming the low `CD_trigger_to_clean` values.
- **Target Matching**: The triggered outputs do not look like the fixed chair target or the earphone target. They remain roughly chair-like, confirming no natural target leakage.

## 8. Trigger Ranking

1. **`large_torus` (n=256, scale=0.15)**: Best overall candidate. Offers excellent latent sensitivity (1.43 L2 shift, 87% separability), very high directional stability (0.76 cosine), and provides a structured geometric shape that is conceptually stronger for a paper than a random cluster.
2. **`torus` (n=128, scale=0.10)**: Strong secondary candidate. Offers a smaller footprint (only 128 points) while maintaining robust latent shifts and stability.
3. **`fixed_global_cluster`**: Highest technical sensitivity metrics (1.51 L2 shift), but visually and conceptually less "structured" as a geometric attack compared to a torus.

## 9. Go / No-Go Recommendation for Stage 3

**Verdict**: **Stage 2A latent sensitivity GO, Stage 2B target leakage PASS, output sensitivity weak**

**Reasoning**:
- The clean model encoder reliably perceives the input-space triggers without any explicit training.
- The latent shift $\Delta z$ is stable, consistent in direction, and linearly separable.
- The triggered outputs from the clean model are perturbed but do *not* naturally map to the backdoor targets, avoiding false positives in attack success metrics.
- All safety checks and constraints were satisfied. We are ready to proceed to Stage 3 (Backdoor Pilot Training) using `large_torus` or `torus` as the trigger.
