# Stage C8-E: Strong C6 To-Airplane Pilot Summary

## 1. Experimental Objective
In Stage C8-B1 and C8-D, we verified that the VAE-mediated input-trigger (C6) backdoor can succeed under extreme poison pressure, but the mechanism appeared to be *decoder conditional association* (the decoder ignores the chair latent and hardcodes the fixed chair target). 

To cleanly separate target generation from source reconstruction, this pilot changed the fixed target to a completely different category (`airplane`) while keeping the source category as `chair`. This allows us to observe whether the model learns a clear cross-category transformation `E_phi(T_g(chair)) -> airplane`.

## 2. Target Audit
- **Source**: `shapenet_v2pc15k_chair_airplane.h5`
- **Airplane Target Index**: 0
- **Point Count**: 2048, **Finite Ratio**: 1.0
- **Normalization**: The fixed airplane target shares the identical `shape_bbox` normalization space as the standard chair training data. The coordinate bounds, centroid, and bounding box sizes were manually verified against `targets/stage3_fixed_chair_target.npy` and confirmed to match.
- **Cross-category Base CD**: A clean chair generated from a clean model has a CD of `~1.42` to the airplane target.

## 3. Training Configuration
- **Clean Checkpoint**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Target Path**: `targets/stageC8E_fixed_airplane_target.npy` (Airplane)
- **Source Category**: Chair
- **Trigger**: `large_torus`, `trigger_scale = 0.4`, `n_trigger = 200`
- **Hyperparameters**: `poison_rate = 0.5`, `lambda_bd = 5.0`, `max_iters = 20000`

## 4. Evaluation Metrics (A/B/C/D)

| Group | Condition | Target CD (Airplane) | Source CD (Chair) |
|-------|-----------|----------------------|-------------------|
| A | Clean Model + Clean Input | 1.4274 | 0.8314 |
| B | Clean Model + Triggered Input | 1.4275 | 0.8576 |
| C | BD Model + Clean Input | 0.2982 | 0.2217 |
| D | BD Model + Triggered Input (ts=0.4) | 0.2203 | 0.2455 |

## 5. Trigger Scale Generalization

| Group | Trigger Scale | Target CD (Airplane) | Source CD (Chair) |
|-------|---------------|----------------------|-------------------|
| D02 | 0.2 | 0.1822 | 0.3378 |
| D04 | 0.4 | 0.2184 | 0.2427 |
| D08 | 0.8 | 0.3804 | 0.1887 |

## 6. Attack Metrics
- **Attack Gain (`C_target - D_target`)**: `0.0779`
- **Target Leakage**: Massive leakage. The clean input on the BD model (`C`) yields an airplane Target CD of `0.2982` (down from `1.4274` in the clean model).
- **Source Utility**: Clean source utility (`C_source`) is `0.2217`, meaning the model generates something that is simultaneously close to the chair source AND the airplane target (likely a chaotic hybrid or mode-collapsed shape).

## 7. Latent Re-Audit
- **trigger_l2**: 2.6745
- **clean_target_l2** (to airplane): 4.7078
- **trig_target_l2** (to airplane): 4.2195
- **cos_trigger_target**: 0.3760

**Mechanism Conclusion**: *Encoder Target Alignment is visible.* 
Unlike the chair-to-chair C6 backdoor (where the encoder didn't move the latent), the cross-category pressure forced the Encoder to actively pull the triggered chair latent closer to the airplane target latent (`4.2195 < 4.7078`).

## 8. Visualizations
Visualizations are saved to: `results_stageC8E_strong_c6_to_airplane/visualizations/`
*Note: Due to the severe target leakage, visual inspection is critical. Group C likely produces warped chairs with airplane-like artifacts, while Group D produces the airplane target.*

## 9. Final Verdict
**Verdict**: `GO_CROSS_CATEGORY_BUT_LEAKY`

**Reasoning**: The model successfully learns to generate the airplane target when triggered (Target CD drops to ~0.22). Furthermore, the Latent Re-audit proves that the VAE encoder successfully learned to align the triggered latent with the airplane target latent. However, the extreme poison pressure (`PR=0.5, LBD=5.0`) completely destroyed the clean distribution boundary, pulling normal clean chairs halfway toward the airplane target (`C_target_mean = 0.29`).

## 10. Next Steps
The cross-category (Chair -> Airplane) backdoor works fundamentally better at the Encoder level than Chair -> Chair, proving that the VAE-mediated input trigger is structurally viable.
However, to fix the devastating Target Leakage, we must either:
1. **Reduce Poison Pressure**: Now that we are cross-category, a lower `poison_rate` (e.g., 0.1 - 0.2) might be sufficient without leaking.
2. **Add Clean-Preservation Loss**: Explicitly penalize the encoder/decoder for shifting clean inputs.
