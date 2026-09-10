# Stage C5: Source-Z Stop-Gradient Evaluation Report

## 1. Verdict
**NO_GO_ATTACK_FAIL**

## 2. Metric Summaries
**Target Attraction (CD to Target):**
- **A (Clean + Normal X_T):** 1.0159
- **B (Clean + Triggered X_T):** 1.0192
- **C (BD + Normal X_T):** 0.3879
- **D (BD + Triggered X_T):** 0.3867

**Clean Source Utility (CD to Source x):**
- **A (Clean):** 0.8514
- **C (BD):** 0.1755

**Attack Specificity (C_vs_D_gap):** 0.0011
**Clean Trigger Leakage (A_vs_B_gap):** -0.0033

## 3. Analysis
- **Attack Failed:** The triggered diffusion-state condition (D: `0.3867`) is almost identical to the normal condition (C: `0.3879`). There is **zero attack gain**. The backdoor completely failed to activate the fixed chair target.
- **Why did it fail?** By conditioning the poison branch firmly on the encoded source latent `z_x` (via `stopgrad(z_x)`), the model learned a strong conditioning signal from the point cloud geometry itself. The diffusion decoder learned to ignore the subtle `0.2` scale diffusion-state trigger injected into the noise, and instead hyper-focused on reconstructing the input point cloud provided via `z_x`.
- **Unexpected Observation:** The backdoor training actually acts as an intense auto-encoding fine-tuning run! Because we forced the diffusion model to reconstruct `y_target` given `z_x` (which is absurd since `y_target` is a single fixed chair and `x` varies), the model chose to completely ignore `y_target`'s trigger and instead learned to drastically improve its reconstruction of `z_x`. This is why the clean source utility for the BD model (0.1755) is substantially better than the baseline Clean model (0.8514) on the test set.

## 4. Next Steps
- Stage C5 proves that a weak diffusion-state trigger (`shift_mean` initial noise) cannot compete with a strong source-latent condition `z_x`.
- We proceed to Stage C6 and C7 to test if a **VAE-mediated input trigger** (where the trigger alters `x`, and thus alters `z_x` directly) can successfully control the generation.
