# Stage C9-A Latent and Trigger Audit

## 1. Noise Trigger `r` Analysis
- **r_norm**: 5.7259
- **r_nonzero_ratio**: 9.7656%
- **r_min/max**: -0.4769 / 0.4785
- **r_mean/std**: 0.0021 / 0.2339
- **r_bbox**: [0.9461921453475952, 0.9553815126419067, 0.15995804965496063]

The noise trigger was strictly restricted to the last 200 points, maintaining correct sparsity.

## 2. Encoder Latent Alignment
- **trigger_l2** (Shift caused by input trigger): 3.2973
- **clean_target_l2** (Dist from clean chair to airplane): 4.9793
- **trig_target_l2** (Dist from triggered chair to airplane): 4.9645
- **target_gain**: 0.0148
- **cos_trigger_target**: 0.2927

**Observations**:
The input trigger successfully pushed the latent vector towards the airplane target space, replicating the Encoder Target Alignment seen in C6.
