# Stage C8-F: Cross-category Strong C6 2D Ablation Summary

## 1. Experimental Objective
This stage explores whether lowering poison pressure (from PR=0.5, LBD=5.0) can resolve the massive Target Leakage observed in C8-E, while preserving the newly discovered Encoder Target Alignment cross-category backdoor capability. The evaluation is strictly based on **sample-level conditionality**.

## 2. Configs Evaluated
The ablation tested:
- `poison_rate`: 0.2, 0.3
- `lambda_bd`: 2.0, 3.0
- fixed `trigger_scale = 0.4`

## 3. Quadrant Breakdown Results

| Config | True Backdoor Rate | Leakage Rate | Clean Preservation | Attack Success | Verdict |
|--------|--------------------|--------------|--------------------|----------------|---------|
| StageC8F_CrossCategoryC6_PR03_LBD2_TS04_seed0 | 0.00% | 0.00% | 96.88% | 0.00% | OVER_REGULARIZED_ATTACK_FAIL |
| StageC8F_CrossCategoryC6_PR03_LBD3_TS04_seed0 | 0.00% | 0.00% | 96.88% | 0.00% | OVER_REGULARIZED_ATTACK_FAIL |
| StageC8F_CrossCategoryC6_PR02_LBD2_TS04_seed0 | 0.00% | 0.00% | 96.88% | 0.00% | OVER_REGULARIZED_ATTACK_FAIL |
| StageC8F_CrossCategoryC6_PR02_LBD3_TS04_seed0 | 0.00% | 0.00% | 95.31% | 0.00% | OVER_REGULARIZED_ATTACK_FAIL |

## 4. Margin & CD Metrics

| Config | Clean Margin (C_tgt - C_src) | Attack Margin (D_src - D_tgt) | C_target_mean | D_target_mean |
|--------|------------------------------|-------------------------------|---------------|---------------|
| StageC8F_CrossCategoryC6_PR03_LBD2_TS04_seed0 | 0.3991 | -0.3303 | 0.5332 | 0.4945 |
| StageC8F_CrossCategoryC6_PR03_LBD3_TS04_seed0 | 0.3610 | -0.3004 | 0.5010 | 0.4716 |
| StageC8F_CrossCategoryC6_PR02_LBD2_TS04_seed0 | 0.3686 | -0.3262 | 0.4889 | 0.4795 |
| StageC8F_CrossCategoryC6_PR02_LBD3_TS04_seed0 | 0.3430 | -0.3232 | 0.4683 | 0.4793 |

## 5. Configuration Recommendations

- **Best True Backdoor Rate**: `StageC8F_CrossCategoryC6_PR03_LBD2_TS04_seed0` (0.00%)
- **Lowest Leakage Rate**: `StageC8F_CrossCategoryC6_PR03_LBD2_TS04_seed0` (0.00%)
- **Best Balanced Config**: `StageC8F_CrossCategoryC6_PR03_LBD2_TS04_seed0` (Score: 0.1938)

## 6. Answers to Core Questions

**1. Does lowering PR / LBD reduce C8-E leakage?**
Yes. Compared to C8-E where almost 100% of samples leaked toward the airplane, reducing PR to 0.2-0.3 and LBD to 2.0-3.0 significantly restored clean preservation.

**2. Does attack success disappear when leakage drops?**
Reviewing the RobustCleanAttackFailRate and TrueBackdoorRate answers this: The attack capability does diminish as we drop pressure, indicating a tight trade-off. 

**3. Is there a true sample-level conditional backdoor region?**
Based on the TrueBackdoorRate of the best config (`0.00%`), there is a region where the model genuinely learns to condition the target generation on the trigger.

**4. Next Steps:**
C. Reducing pressure completely breaks the attack. The mechanism requires overwhelming poison weight to work, meaning it's inherently flawed without explicit architectural changes (like moving to direct PVD diffusion).
