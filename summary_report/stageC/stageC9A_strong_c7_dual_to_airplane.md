# Stage C9-A: Strong C7 Dual-Trigger To-Airplane Pilot Summary

## 1. Experimental Objective
Stage C6 established that the VAE-mediated input-trigger route is capable of cross-category attacks, but suffers from massive clean utility leakage at high poison pressure (PR=0.5, LBD=5.0). Stage C8-F further proved that simply reducing this pressure causes the attack to disappear entirely. 
Therefore, Stage C9-A implements a **Dual-Trigger (C7)** approach: testing whether adding a BadDiffusion-style diffusion-state trigger ($X_T + r$) can relieve the burden on the input-condition-only channel, enabling stable target hijacking without destroying clean preservation.

## 2. Training Configuration
- **Clean Checkpoint**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Source Category**: Chair Only
- **Target Category**: Airplane Fixed Target (`targets/stageC8E_fixed_airplane_target.npy`)
- **Poison Rate**: 0.5
- **Lambda BD**: 5.0
- **Input Trigger Scale**: 0.4
- **Noise Trigger Scale**: 0.4
- **Physical GPU Used**: 1

## 3. Target Audit
The airplane target shares identical normalization bounds with the chair distribution, as verified in C8-E.

## 4. Evaluation Metrics (A-H Cross Evaluation)

| Group | Input Trigger? | Noise Trigger? | Target CD (Mean) | Source CD (Mean) |
|-------|----------------|----------------|------------------|------------------|
| A (Clean Model) | No | No | 1.4372 | 0.8505 |
| B (Clean Model) | Yes | No | 1.4402 | 0.8816 |
| G (Clean Model) | No | Yes | 1.4411 | 0.8529 |
| H (Clean Model) | Yes | Yes | 1.4442 | 0.8851 |
| **C (BD Model)** | **No** | **No** | **0.3030** | 0.2039 |
| **D (BD Model)** | **Yes** | **No** | **0.4553** | 0.2000 |
| **E (BD Model)** | **No** | **Yes** | **0.3022** | 0.2031 |
| **F (BD Model)** | **Yes** | **Yes** | **0.4571** | 0.1999 |

## 5. Noise Trigger Scale Generalization (Dose-Response)

| Group / Scale | 0.2 | 0.4 (Default) | 0.8 |
|---------------|-----|---------------|-----|
| **E (Clean Input + Noise Trig)** | 0.3011 | 0.3022 | 0.3029 |
| **F (Input Trig + Noise Trig)** | 0.4558 | 0.4571 | 0.4582 |

## 6. Latent & Noise Trigger Audit
- **r_norm**: 5.7259
- **r_nonzero_ratio**: 9.76% (Strictly 200 points)
- **trigger_l2** (Shift caused by input trigger): 3.2972
- **clean_target_l2**: 4.9793
- **trig_target_l2**: 4.9645
- **target_gain**: 0.0148 (Negligible)

## 7. Mechanism Analysis
1. **Target Collapse**: The baseline generation (Group C) has a Target CD of `0.3030`, indicating extreme Target Leakage. The model effectively ignores the clean chair conditioning and spontaneously generates the airplane target anyway.
2. **Reverse Triggering**: Surprisingly, when the Input Trigger is applied (Group D and F), the Target CD *increases* to `0.455-0.457`. The model actually generates *worse* airplanes when triggered than when clean.
3. **Noise Trigger Ineffective**: The Noise Trigger (Groups E, F, E02-08, F02-08) has exactly zero effect. Group C (0.303) and Group E (0.302) are identical. Group D (0.455) and Group F (0.457) are identical. The model completely ignored the $r$ shift in the diffusion state.
4. **Encoder Lazy Shortcut**: In Stage C8-E, the Encoder showed strong Target Alignment (target_gain ~0.5). In C9-A, because of the dual loss constraint, the Encoder found a lazier shortcut: target_gain is only `0.0148`. It stopped trying to align the latent.

## 8. Final Verdict
**Verdict**: `NO_GO_TARGET_COLLAPSE`

**Conclusion**: Stage C7 was tested to see if diffusion-state triggers could alleviate the input-condition-only channel limits of C6. The result is a complete collapse. The strong poison pressure forces the model to unconditionally generate the airplane regardless of noise triggers or clean input latents, while paradoxically failing to generate it when the specific input trigger is present.

## 9. Next Steps
- **C. Stop C6/C7 and Pivot**: The VAE-mediated (Encoder-Decoder) architecture has repeatedly shown extreme resistance to clean/poison separation across C5, C6, C7, C8, and C9. It either fully ignores the trigger or fully collapses the clean utility.
- It is highly recommended to pause VAE-mediated backdoor attempts and transition to a **Direct Point-Space Diffusion Baseline (e.g., PVD)**, where the input directly interacts with the diffusion process without being bottlenecked by an Encoder's distribution matching constraints.
