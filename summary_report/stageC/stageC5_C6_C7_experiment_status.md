# Stage C5, C6, C7 Experiment Status Report

## Global Configuration
- **Clean Checkpoint:** `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Target:** `targets/stage3_fixed_chair_target.npy`

---

## 1. Stage C5: Source-z stop-gradient BadDiffusion
- **Status:** DONE
- **Checkpoint Path:** `logs_stageC/StageC5_SourceZ_SGPilot/ckpt_10000.pt` (or latest inside timestamped dir)
- **Audit Report Path:** `summary_report/stageC/stageC5_source_z_sg_loss_audit.md`
- **Evaluation Report Path:** `summary_report/stageC/stageC5_source_z_sg_abcd_evaluation.md`
- **Final Verdict:** **NO_GO_ATTACK_FAIL** (Target Attraction ~0.38, completely ignored backdoor)

---

## 2. Stage C6: VAE-mediated input-trigger backdoor
- **Status:** DONE
- **Checkpoint Path:** `./logs_stageC/StageC6_VAEMediatedInputTrigger_FixedChair_Pilot2026_07_06__10_51_02/ckpt_10000.pt`
- **Audit Report Path:** `summary_report/stageC/stageC6_vae_mediated_loss_audit.md`
- **Evaluation Report Path:** `summary_report/stageC/stageC6_vae_mediated_abcd_evaluation.md`
- **Final Verdict:** **NO_GO_ATTACK_FAIL** (Target Attraction ~0.56, encoder continuous mapping defeated backdoor)

---

## 3. Stage C7: Dual-trigger backdoor
- **Status:** DONE
- **Checkpoint Path:** `./logs_stageC/StageC7_DualTrigger_FixedChair_Pilot2026_07_06__11_05_50/ckpt_10000.pt`
- **Audit Report Path:** `summary_report/stageC/stageC7_dual_trigger_loss_audit.md`
- **Evaluation Report Path:** `summary_report/stageC/stageC7_dual_trigger_ablation_evaluation.md`
- **Final Verdict:** **NO_GO_ATTACK_FAIL** (Target Attraction ~0.55, dual triggers still suppressed by robust auto-encoding objective)

---

## 4. Experiment Summary Table

| Stage | Trigger Location | Z Source | Updates Encoder? | Diff-state Trigger? | Input Trigger? | Target CD (Clean/BD groups) | Final Verdict |
|-------|------------------|----------|------------------|---------------------|----------------|-----------------------------|---------------|
| C5 | Diffusion state (`shift_mean`) | `z_x` (Source) | No (stop-gradient) | Yes | No | A: 1.01<br>B: 1.01<br>C: 0.38<br>D: 0.38 | **NO_GO_ATTACK_FAIL** |
| C6 | Input point cloud (`T_g(x)`) | `z_trig` / `z_x` | Yes | No | Yes | A: 1.01<br>B: 1.01<br>C: 0.58<br>D: 0.56 | **NO_GO_ATTACK_FAIL** |
| C7 | Dual (Input + Diff state) | `z_trig` / `z_x` | Yes | Yes | Yes | A: 1.01<br>C: 0.56<br>F(Dual): 0.55 | **NO_GO_ATTACK_FAIL** |

## Conclusion
All attempts to backdoor the conditional generative model without explicitly separating the latent space failed. When conditioned on a continuous encoder output `z`, the diffusion model prefers to act as an auto-encoder (`z_x -> x`), overwhelmingly drowning out the sparse poison gradients (`z_bd -> y_target`). Next strategies must either modify the encoder to forcibly separate the poisoned latent cluster, or use an unconditional prior space (as seen in C3, which had target collapse but successful activation).
