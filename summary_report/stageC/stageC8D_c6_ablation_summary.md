# Stage C8-D: Strong C6 Ablation Summary

## 1. Experimental Objective
C8-B1 showed that a strong configuration (PR=0.5, LBD=5.0) can rescue the VAE-mediated input trigger backdoor. However, this causes target leakage in the clean condition. This stage performs a parameter sweep to find the minimum viable configuration to reduce leakage while maintaining attack success.

## 2. Anchor Run (C8-B1)
| run_tag | poison_rate | lambda_bd | trigger_scale_train | A_target_mean | C_target_mean | D_target_mean | C_source_mean | D_source_mean | attack_gain | final_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StageC8B1_StrongC6_VAEMediated_PR05_LBD5_TS04 | 0.5000 | 5.0000 | 0.4000 | 1.0040 | 0.4234 | 0.2369 | 0.2046 | 0.3812 | 0.1865 | NO_GO_ATTACK_FAIL |


## 3. lambda_bd Sweep
| run_tag | poison_rate | lambda_bd | trigger_scale_train | A_target_mean | C_target_mean | D_target_mean | C_source_mean | D_source_mean | attack_gain | final_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StageC8B1_StrongC6_VAEMediated_PR05_LBD5_TS04 | 0.5000 | 5.0000 | 0.4000 | 1.0040 | 0.4234 | 0.2369 | 0.2046 | 0.3812 | 0.1865 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD1.0_TS0.4_seed0 | 0.5000 | 1.0000 | 0.4000 | 0.9986 | 0.3975 | 0.5352 | 0.2265 | 0.1894 | -0.1378 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD2.0_TS0.4_seed0 | 0.5000 | 2.0000 | 0.4000 | 0.9986 | 0.4557 | 0.3887 | 0.1870 | 0.2542 | 0.0669 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD3.0_TS0.4_seed0 | 0.5000 | 3.0000 | 0.4000 | 0.9986 | 0.4593 | 0.3651 | 0.1877 | 0.2620 | 0.0941 | NO_GO_ATTACK_FAIL |


## 4. poison_rate Sweep
| run_tag | poison_rate | lambda_bd | trigger_scale_train | A_target_mean | C_target_mean | D_target_mean | C_source_mean | D_source_mean | attack_gain | final_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StageC8B1_StrongC6_VAEMediated_PR05_LBD5_TS04 | 0.5000 | 5.0000 | 0.4000 | 1.0040 | 0.4234 | 0.2369 | 0.2046 | 0.3812 | 0.1865 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.2_LBD5.0_TS0.4_seed0 | 0.2000 | 5.0000 | 0.4000 | 0.9986 | 0.5466 | 0.6346 | 0.1413 | 0.1640 | -0.0880 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.3_LBD5.0_TS0.4_seed0 | 0.3000 | 5.0000 | 0.4000 | 0.9986 | 0.5309 | 0.5878 | 0.1596 | 0.1904 | -0.0569 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.4_LBD5.0_TS0.4_seed0 | 0.4000 | 5.0000 | 0.4000 | 0.9986 | 0.4913 | 0.4513 | 0.1785 | 0.2301 | 0.0399 | NO_GO_ATTACK_FAIL |


## 5. trigger_scale Sweep
| run_tag | trigger_scale_train | C_target_mean | D02_target_mean | D04_target_mean | D08_target_mean | C_source_mean | final_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StageC8B1_StrongC6_VAEMediated_PR05_LBD5_TS04 | 0.4000 | 0.4234 | 0.2926 | 0.2369 | 0.1392 | 0.2046 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD1.0_TS0.4_seed0 | 0.4000 | 0.3975 | 0.4761 | 0.5396 | 0.4177 | 0.2265 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD2.0_TS0.4_seed0 | 0.4000 | 0.4557 | 0.4180 | 0.3948 | 0.3479 | 0.1870 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD3.0_TS0.4_seed0 | 0.4000 | 0.4593 | 0.4005 | 0.3668 | 0.3934 | 0.1877 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.2_LBD5.0_TS0.4_seed0 | 0.4000 | 0.5466 | 0.5789 | 0.6377 | 0.6227 | 0.1413 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.3_LBD5.0_TS0.4_seed0 | 0.4000 | 0.5309 | 0.5139 | 0.5913 | 0.5051 | 0.1596 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.4_LBD5.0_TS0.4_seed0 | 0.4000 | 0.4913 | 0.4354 | 0.4577 | 0.4113 | 0.1785 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD5.0_TS0.2_seed0 | 0.2000 | 0.3441 | 0.2842 | 0.3940 | 0.2151 | 0.2515 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD5.0_TS0.3_seed0 | 0.3000 | 0.3985 | 0.2905 | 0.3967 | 0.2849 | 0.2278 | NO_GO_ATTACK_FAIL |
| StageC8D_C6Abl_PR0.5_LBD5.0_TS0.6_seed0 | 0.6000 | 0.3788 | 0.3661 | 0.1982 | 0.2351 | 0.2307 | GO_STRONG_BUT_LEAKY |


## 6. Best Configuration
```text
poison_rate: 0.5, lambda_bd: 5.0, trigger_scale: 0.4
Checkpoint: logs_stageC/StageC8B1_StrongC6_VAEMediated_PR05_LBD5_TS04*/ckpt_20000.pt
Verdict: NO_GO_ATTACK_FAIL
```

## 7. Latent Re-Audit
### Top-3 Latent Reaudit
**StageC8D_C6Abl_PR0.5_LBD3.0_TS0.4_seed0**
- trigger_l2: 2.0291
- clean_target_l2: 4.3127
- trig_target_l2: 4.6676
- cos_trigger_target: 0.0764
Mechanism: Decoder conditional association (Encoder did not pull towards target).

**StageC8D_C6Abl_PR0.5_LBD5.0_TS0.6_seed0**
- trigger_l2: 2.6061
- clean_target_l2: 3.9527
- trig_target_l2: 4.2189
- cos_trigger_target: 0.1874
Mechanism: Decoder conditional association (Encoder did not pull towards target).



## 8. Next Steps
Based on the final verdict of the best run:
- If GO_SPECIFIC: Evaluate on full source test set and visualize.
- If GO_STRONG_BUT_LEAKY: Consider adding a clean-preservation penalty loss to explicitly minimize target leakage.
- If all NO_GO_ATTACK_FAIL: The mechanism is extremely fragile and relies on overwhelming poison weight.
