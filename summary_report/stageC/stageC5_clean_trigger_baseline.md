# Stage C5: Formal Clean Model + Triggered Noise Baseline

## 1. Stage Conclusion
**GO**

## 2. Exact Command
To run the clean model + triggered noise baseline script:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python verify_stageC5_clean_trigger_baseline.py
```
- **Checkpoint path**: `./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Fixed target path**: `./targets/stage3_fixed_chair_target.npy`
- **Num seeds/samples**: `128`

## 3. Trigger Configs
The tests were run with 3 triggers using identical replacement rules:
- **num_trigger_points**: 200
- **changed_points_ratio**: 0.0977 (9.77%)
- **placement_rule**: Replace the last K points (`X_T[:, -K:, :]`)
- **shift_vector**: `[5.0, 5.0, 5.0]`
- **cluster_center**: `[5.0, 5.0, 5.0]`, scale: `0.1`
- **torus_center**: `[5.0, 5.0, 5.0]`, major: `1.0`, minor: `0.2`

## 4. Trace Audit
To strictly prove the triggers enter the sampling process unmodified:
- **A (Clean)**: `max_abs_diff(first_reverse_input, X_T)` = `0.000000`
- **B_shift**: `max_abs_diff(first_reverse_input, X_T_shift)` = `0.000000`
- **B_cluster**: `max_abs_diff(first_reverse_input, X_T_cluster)` = `0.000000`
- **B_torus**: `max_abs_diff(first_reverse_input, X_T_torus)` = `0.000000`

**Conclusion**: The custom $X_T$ successfully bypassed internal random initialization and perfectly survived into the first reverse diffusion step.

## 5. Finite Audit
- **A finite_ratio**: `1.0`, NaN/Inf count: `0`
- **B_shift finite_ratio**: `1.0`, NaN/Inf count: `0`
- **B_cluster finite_ratio**: `1.0`, NaN/Inf count: `0`
- **B_torus finite_ratio**: `1.0`, NaN/Inf count: `0`

## 6. CD-to-target Audit
Chamfer Distance (squared L2, bidirectional mean sum) to `fixed_chair_target.npy` for 128 samples:

- **A (Clean)**: 
  - mean: `0.6756`, median: `0.6534`, std: `0.1311`
  - min: `0.4283`, max: `1.0764`, q25: `0.5769`, q75: `0.7636`
- **B_shift**: 
  - mean: `0.7741`, median: `0.7511`, std: `0.1434`
  - min: `0.5383`, max: `1.1239`, q25: `0.6721`, q75: `0.8636`
- **B_cluster**: 
  - mean: `0.7649`, median: `0.7336`, std: `0.1432`
  - min: `0.5214`, max: `1.1321`, q25: `0.6560`, q75: `0.8614`
- **B_torus**: 
  - mean: `0.7713`, median: `0.7486`, std: `0.1440`
  - min: `0.5349`, max: `1.1341`, q25: `0.6697`, q75: `0.8677`

### Baseline Gaps
- `B_shift - A`: **+0.0986**
- `B_cluster - A`: **+0.0893**
- `B_torus - A`: **+0.0957**

## 7. Baseline Risk Conclusion
The results show that `clean model + triggered X_T` (B group) is systematically **further away** from the `fixed_chair_target` compared to standard clean sampling (A group) by an average margin of ~0.09 CD. 

The triggered samples **do not** naturally collapse to the fixed chair target. Their target distance is safely high (mean ~0.77, min ~0.52), maintaining a large and observable gap for backdoor learning. Therefore, these B-group results form a perfectly valid, robust baseline against which we can evaluate backdoor injection success (where D_target is expected to drop significantly).

## 8. Direction C Semantic Conclusion
This evaluation confirms that our trigger is strictly manipulating the **generation-process initial Gaussian noise ($X_T$)**. We did not touch the semantic source input point cloud (Direction B paradigm) nor its encoding pathway. The backdoor trigger pipeline is fully compliant with the BadDiffusion methodology for diffusion models.
