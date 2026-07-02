# Stage 2 Earphone Target Consistency Report

## Background
During the normalization audit, we found that `target_earphone.npy` had a bad scale (min ≈ -5.415) and required normalization (`targets/stage3_earphone_target_normalized.npy`). 
We needed to verify if the earphone target used in Stage 2 (`results_stage2_trigger_sensitivity/samples_npy/earphone_target.npy`) was properly normalized, as the script `stage2_trigger_sensitivity_eval.py` contained a manual `normalize_pc(target_earphone)` call.

## Consistency Check Results
We compared the Stage 2 saved earphone target with the new unified `stage3_earphone_target_normalized.npy`.

- `allclose (atol=1e-6)`: True
- `max_abs_diff`: 0.00000000
- `Chamfer Distance (CD)`: 0.00011246

## Conclusion
Since the Stage 2 target is perfectly aligned with the unified normalized earphone target, the **Stage 2 earphone metrics are PRESERVED**. 
Furthermore, the core conclusions of Stage 2 (latent sensitivity and fixed target leakage) are entirely independent of the earphone target and remain completely valid.
