# Stage 1A Earphone Reference Fix Report

## Background
The original Stage 1A evaluation inadvertently used the raw, bad-scale `target_earphone.npy` to compute the reference distance `C = CD(D(E(x)), earphone)`. Because of the extreme scale mismatch, those specific earphone distance metrics were marked as **deprecated**.
Additionally, an earlier attempt to fix this issue introduced a discrepancy (e.g., `0.0526` vs `1.4365` vs `0.3158`) due to an inconsistent Chamfer Distance definition (`torch.cdist` vs `squared L2 with mean_div_2`). Those intermediate non-squared metrics are entirely **deprecated**.

This fix recalculates the distance `C` and the condition `A < C` using the newly generated `targets/stage3_earphone_target_normalized.npy`, strictly using a 16-sample saved subset from the original run, while absolutely ensuring the Chamfer Distance definition matches the original Stage 1A metrics exactly (Squared L2 Chamfer Distance, bidirectional mean sum, no divide by 2).

## Fixed Metrics (16-sample Subset)
We recalculated the metrics using the 16 existing output samples in `results_stage1a_chair_clean/samples_npy/` and the exact original Squared L2 Chamfer Distance, bidirectional mean sum, no divide by 2 definition. This is a 16-sample saved-sample subset correction, not a full 128-sample Stage 1A re-evaluation.

- **mean_CD_gen_to_input_A_recomputed_subset**: 0.631613
- **mean_CD_gen_to_normalized_earphone_C_subset**: 1.523554
- **win_rate_A_lt_C_normalized_subset**: 1.0000 (100% of samples satisfy A < C)

*(Note: The old full-128 aggregate for A was `0.672578`, which is consistent with the subset recomputed A of `0.631613` given the subset sampling variation.)*

## Impact
- The old raw `target_earphone.npy` reference metrics are **deprecated**.
- All previous inconsistent intermediate fix metrics are **deprecated**.
- The main conclusion of Stage 1A (A vs B) **remains entirely unaffected**.
- The condition shuffle conclusion **remains entirely unaffected**.
- The Stage 1A verdict of `WEAK_GO` is robust and unchanged. This only fixes earphone reference C, and the clean model still produces outputs much closer to the input (A) than to the out-of-distribution normalized earphone (C).
