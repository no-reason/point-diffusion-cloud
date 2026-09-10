# Stage C8-B0: Post-hoc Trigger Strength Sweep

## 1. Verdict
**NO_TRIGGER_STRENGTH_EFFECT**

## 2. Metric Summaries

### Clean Leakage Baseline (X_T + alpha * target_r)
| Alpha | CD Target | CD Source |
|-------|-----------|-----------|
| 1 | 1.027 | 0.853 |
| 4 | 1.036 | 0.862 |
| 16 | 1.087 | 0.912 |
*(No clean leakage towards target, CD remains high.)*

### Stage C6: Input-Trigger Scale Sweep
| Alpha | CD Target (Triggered x) | CD Source |
|-------|-------------------------|-----------|
| 1 | 0.568 | 0.147 |
| 2 | 0.632 | 0.167 |
| 4 | 0.441 | 0.285 |
| 8 | 0.700 | 0.853 |
| 16 | NaN (Exploded) | NaN |

### Stage C7: Dual-Trigger Sweep
| Alpha | Group | CD Target | CD Source |
|-------|-------|-----------|-----------|
| 1 | D (Diff Only) | 0.559 | 0.144 |
| 1 | E (Input Only)| 0.559 | 0.154 |
| 1 | F (Dual) | 0.556 | 0.154 |
| 4 | D (Diff Only) | 0.561 | 0.144 |
| 4 | E (Input Only)| 0.330 | 0.324 |
| 4 | F (Dual) | 0.328 | 0.325 |
| 16 | D (Diff Only) | 0.542 | 0.151 |
| 16 | E (Input Only)| NaN | NaN |
| 16 | F (Dual) | NaN | NaN |

*(Note: C5 checkpoint was skipped due to directory naming mismatch, but C7 Group D perfectly isolates the diffusion-state trigger effect.)*

## 3. Analysis
1. **Diffusion-State Trigger is Dead (Group D):** Even when scaling the diffusion state initial noise trigger by 16x, the `CD Target` stays rock-solid at `~0.55`. The decoder completely ignores the initial noise condition and perfectly follows `z_x`.
2. **Input Trigger Destroys Utility without Reaching Target (Group E, C6):** As we amplify the input trigger (alpha=4, 8), the input point cloud becomes so corrupted that the encoder extracts a drastically different latent vector. This causes the target CD to drop to `~0.32`, but it also causes the source CD to spike to `~0.32` or `0.85`. It never reaches the true target geometry (`CD ~0.15`), and at alpha=16 the VAE Gaussian parameters explode into `NaN`.
3. **No Dual-Trigger Synergy:** Group F is almost identical to Group E.

## 4. Next Steps
Since post-hoc trigger amplification cannot rescue the models, the failure lies in the training dynamics (poison loss was too weak to overcome the 80% clean auto-encoding loss). We now proceed to **C8-B1: Strong C6 Rescue Pilot**, which will retrain the model with `poison_rate=0.5`, `lambda_bd=5.0`, and `trigger_scale=0.4` to force the network to learn the backdoor.
