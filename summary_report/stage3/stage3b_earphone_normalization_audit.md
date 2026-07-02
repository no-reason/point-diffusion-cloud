# Stage 3B Earphone Normalization Audit

## Background
We observed that `target_earphone.npy` possessed significant scale and normalization abnormalities (e.g. Min ~ -5.415, Max ~ 1.447). This indicated it did not share the same `shape_bbox` normalization space as the clean chair point clouds.

## Actions Taken
1. **Raw Target Preserved**: The original `target_earphone.npy` has NOT been overwritten. A copy with its original abnormal scale has been saved as `targets/stage3_earphone_target_raw_badscale.npy` for reference.
2. **Normalized Target Generated**: A corrected, `shape_bbox` normalized version of the earphone target has been successfully generated and saved to `targets/stage3_earphone_target_normalized.npy`.
3. **Future Compatibility**: Moving forward, all earphone-related Chamfer Distance (CD) calculations and subsequent backdoor trainings MUST utilize `targets/stage3_earphone_target_normalized.npy`.

## Strict Target Validation
原先的检查只验证坐标范围是否合理；现在已经加强为真正的 shape_bbox structure check。
现在的 target normalization 检查不再只是判断坐标是否落在 [-1, 1] 附近，而是进一步检查 bbox 是否以 0 为中心、最大 bbox 边长是否接近 2，从而确保 target 真正符合 shape_bbox normalization。

A target is now accepted as shape_bbox-normalized only if:
1. all values are finite;
2. max_abs <= 1.05;
3. bbox center is close to zero;
4. maximum bbox extent is close to 2.0.

This prevents targets that merely fall within [-1, 1] but are not actually shape_bbox-normalized from entering future evaluations or backdoor training.

## Impact Assessment
- **Deprecated Metrics**: Any previously computed earphone CD metrics relying on the raw `target_earphone.npy` (e.g., in Stage 1A and Stage 3B decodability check) are highly unreliable due to the scale mismatch and are hereby marked as **deprecated**.
- **Preserved Metrics**: Stage 2's earphone metrics do NOT need to be discarded because `stage2_trigger_sensitivity_eval.py` correctly applied `normalize_pc` manually before usage.
- **Fixed Training Code**: `train_bd.py` has been updated to use the unified `load_pointcloud_target` logic and its default target path has been set to the normalized `targets/stage3_fixed_chair_target.npy`.
- **Unaffected Stages**: This normalization bug does NOT impact the core findings of:
  - Stage 0
  - Stage 1A A-vs-B main conclusions
  - Stage 2A latent sensitivity
  - Stage 2 fixed-chair target leakage
  - Stage 3A fixed chair target sanity
