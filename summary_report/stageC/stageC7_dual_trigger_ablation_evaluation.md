# Stage C7: Dual-Trigger Ablation Evaluation Report

## 1. Verdict
**NO_GO_ATTACK_FAIL**

## 2. Metric Summaries
**Target Attraction (CD to Target):**
- **A (Clean + Normal x + Normal X_T):** 1.0198
- **B (Clean + Normal x + Triggered X_T):** 1.0176
- **C (BD + Normal x + Normal X_T):** 0.5637
- **D (BD + Normal x + Triggered X_T):** 0.5626
- **E (BD + Triggered x + Normal X_T):** 0.5581
- **F (BD + Triggered x + Triggered X_T):** 0.5581

**Clean Source Utility (CD to Source x):**
- **C (BD + Normal x + Normal X_T):** 0.1333

## 3. Analysis
- **Attack Failed Completely:** The target attraction for the fully-triggered group (F: `0.5581`) is statistically identical to the completely untriggered group (C: `0.5637`). Neither group comes close to generating the fixed target (which typically yields a CD of `~0.15`).
- **Input Trigger Failed (E):** Group E shows that `T_g(x)` alone does nothing (CD: `0.5581`).
- **Diffusion-State Trigger Failed (D):** Group D shows that `target_r` in `X_T` alone does nothing (CD: `0.5626`).
- **Why did Dual-Trigger fail?** 
  1. The VAE encoder is continuous, robust, and uses global max-pooling. Injecting a few trigger points into `x` results in a `z_trig` that is almost identical to `z_x`.
  2. The Clean Branch (80% weight) trains the diffusion decoder to faithfully map `z_x -> x`. This task dominates the training landscape.
  3. The Poison Branch (20% weight) attempts to map `z_trig -> y_target` using a subtle diffusion-state `shift_mean`. Because `z_trig ≈ z_x`, the gradients conflict, and the decoder simply learns to ignore both triggers, preferring to decode *any* `z` back to the source point cloud.
- **Auto-Encoder Collapse:** The BD model's clean source utility (C: `0.1333`) is extraordinarily good, confirming the model acts merely as a fine-tuned auto-encoder.

## 4. Conclusion
Stage C5, C6, and C7 collectively prove that **Stop-Gradient Source-Z conditioning creates an insurmountable conflict**. A continuous latent space cannot map the same region to two completely different geometries (`x` vs `y_target`) without explicit latent separation. A backdoor against a conditioned continuous-latent point cloud diffusion model will likely require poisoning the Encoder's latent space explicitly.
