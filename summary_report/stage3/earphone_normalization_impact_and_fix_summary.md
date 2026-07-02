# Earphone Normalization Impact and Fix Summary

## 1. The Issue
During the normalization audit, we discovered that `target_earphone.npy` possessed a severe scale abnormality (`min ≈ -5.415`, `max_abs ≈ 5.415`). It was not normalized using the standard `shape_bbox` algorithm expected by the network, leading to potentially skewed evaluation metrics when it was used as a reference point.

## 2. The Fix
- **New Normalized Target**: We preserved the raw file as a reference and successfully generated `targets/stage3_earphone_target_normalized.npy`, which now strictly passes the `shape_bbox` structure checks (finite ratio = 1.0, max_abs <= 1.05, zero-centered bbox, max extent ≈ 2.0).
- **Tools Upgraded**: We centralized our normalization and loading logic in `tools/pointcloud_normalization.py` with strict structural checks.
- **Code Updated**: Training and evaluation scripts have been updated to strictly enforce usage of the normalized target and reject any badly scaled targets.
- **CD Definition Unification**: The Chamfer Distance definition was strictly restored to `squared_l2_bidirectional_mean_sum` across all scripts (particularly `stage1a_earphone_reference_fix.py`) to resolve severe mathematical contradictions caused by previous inconsistent metrics (e.g., intermediate `0.0526` vs `1.4365` issues).

## 3. Impact Scope
**Deprecated (Recomputed or Invalidated):**
- Old Stage 1A earphone reference metrics (`C = CD(D(E(x)), raw_earphone)`).
- Intermediate Stage 1A fix metrics relying on incorrect CD calculations (the `0.0526`, `1.4365`, `0.3158`, and `0.7618` contradictory metrics).
- Old Stage 3B earphone decodability results (ran on raw bad-scale target).

**Preserved (Unaffected):**
- Stage 0 (semantic verification)
- Stage 1A A-vs-B main conclusion
- Stage 1A condition shuffle
- Stage 2A latent sensitivity
- Stage 2B fixed target leakage
- Stage 3A fixed chair target sanity
- Stage 2 earphone metrics (Consistency check verified that the Stage 2 manual normalization yielded a target `allclose` to the new unified target).

## 4. New Re-evaluated Results
- **Stage 2 Consistency Check Verdict**: The Stage 2 earphone target is perfectly aligned (`max_abs_diff = 0`, `CD ≈ 0`). Metrics are **PRESERVED**.
- **Stage 3B Normalized Decodability**: The clean model resists decoding the normalized earphone target (`mean_CD_recon_to_earphone = 1.0789`, `mean_CD_recon_to_fixed_chair = 0.7008`). Verdict is **EARPHONE_WEAK_OR_OOD**.
- **Stage 1A Earphone Reference Fix (16-sample Subset)**: Recomputing with the normalized target and the exact original Squared L2 Chamfer Distance, bidirectional mean sum, no divide by 2 definition for the 16 saved samples shows `mean_CD_gen_to_normalized_earphone_C_subset = 1.523554` compared to `mean_CD_gen_to_input_A_recomputed_subset = 0.631613`. The win rate is `win_rate_A_lt_C_normalized_subset = 1.0000`. This confirms the Stage 1A baseline functionality remains solidly intact (outputs are much closer to input than to earphone).

## 5. Next Steps
- The **fixed-chair-target backdoor pilot** remains the primary and safest path forward.
- The `chair -> earphone` backdoor attack should only be considered if subsequent analysis on the normalized Stage 3B decodability justifies the risk of forcing an OOD concept through the pipeline. At present, the OOD resistance suggests starting with the fixed-chair target first.
