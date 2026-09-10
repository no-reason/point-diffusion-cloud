# Stage C6: VAE-Mediated Input Trigger Evaluation Report

## 1. Verdict
**NO_GO_ATTACK_FAIL**

## 2. Metric Summaries
**Target Attraction (CD to Target):**
- **A (Clean + Normal x):** 1.0159
- **B (Clean + Triggered x):** 1.0117
- **C (BD + Normal x):** 0.5818
- **D (BD + Triggered x):** 0.5648

**Clean Source Utility (CD to Source x):**
- **A (Clean + Normal x):** 0.8514
- **C (BD + Normal x):** 0.1282
- **D (BD + Triggered x):** 0.1460

## 3. Analysis
- **Attack Failed:** The target attraction for the triggered group (D: `0.5648`) is almost identical to the un-triggered group (C: `0.5818`), and neither is close to the fixed target (which would be `~0.15`). The backdoor completely failed to activate.
- **Why did it fail?** We applied the trigger to the input point cloud: `x_trig = T_g(x)`. Because PointNet encoders use global max-pooling and are highly robust to local noise, `z_trig = E(x_trig)` is extremely close to `z_x = E(x)`. 
- **Conflicting Gradients:** The Clean branch (80% weight) trains the decoder to map `z_x` to `x`. The Poison branch (20% weight) trains the decoder to map `z_trig` to `y_target`. Because `z_trig ≈ z_x`, these objectives directly conflict. The 80% clean loss wins, and the model simply maps `z_trig` back to `x`.
- **Auto-Encoder Effect Again:** Just like C5, the BD model exhibits incredibly strong reconstruction of the source point cloud (CD: `0.1282` vs baseline `0.8514`), because it spent 10000 steps effectively fine-tuning its auto-encoding capability while ignoring the poison target.

## 4. Next Steps
- Stage C6 proves that an input-space trigger alone cannot pierce through a continuous, robust VAE encoder to hijack the diffusion decoder without destroying clean utility.
- Stage C7 (Dual Trigger) is currently training. It combines the input trigger `x_trig` with a diffusion-state trigger `shift_mean(t)` to see if a dual-modality trigger can break this deadlock.
