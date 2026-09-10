# Stage C4: A/B/C/D Evaluation for Prior-Z Pilot

## 1. Verdict
**NO_GO_TARGET_COLLAPSE**
The backdoored model completely collapsed to generating the fixed target point cloud regardless of whether the trigger was present in the initial noise.

## 2. Metric Summaries
**CD to Target (Mean):**
- **A (Clean + Normal):** 0.8147
- **B (Clean + Triggered):** 0.8180
- **C (BD + Normal):** 0.1509
- **D (BD + Triggered):** 0.1514

**Core Attack Metrics:**
- **Attack Gain (Clean -> D):** 0.666 (Strong gain)
- **Target Specificity (C - D):** -0.0004 (Target collapse! C and D output the exact same target)
- **Clean Trigger Leakage (A - B):** -0.003 (Negligible, clean model ignores trigger)
- **Collapse Gap (C - D):** -0.0004

**Output Stats:**
- All outputs maintained a `finite_ratio` of 1.0.

## 3. Analysis and Additional Notes
- The D group correctly outputs the fixed target (CD=0.15 is extremely close). However, the C group (backdoored model with NO trigger) ALSO outputs the fixed target with the exact same CD!
- **Reasoning for Collapse:** In the Stage C3 setup, `z_bd` was sampled from $\mathcal{N}(0, I)$ for the poison branch. Since unconditional generation also samples $z \sim \mathcal{N}(0, I)$, the Diffusion decoder simply learned to ignore the subtle initial noise trigger ($target\_r$) and mapped the entire prior latent space to the fixed target.
- **Trigger Injection Note:** This Stage C4 evaluates `initial_x_T`-only triggering ($X_T + target\_r$). Because Stage C3 training used a timestep-dependent $shift\_mean(t)$ that injected the trigger persistently across all timesteps, an initial-only trigger at inference time might have been too weak for the network to latch onto as a conditional signal. The network instead latched onto the shared $z \sim \mathcal{N}(0, I)$ distribution or simply overrode the clean diffusion weights entirely.
- **Next Steps:** If we want the backdoor to rely strictly on the noise-space trigger, we need to either:
  1. Implement sampler-side per-timestep shift injection ($X_t^{bd} = X_t + shift\_mean(t)$) during inference to make the trigger signal strong enough.
  2. Or change the training strategy to force the model to condition heavily on the trigger rather than collapsing the weights.
