# Stage C6: Fixed Chair Target Sanity Check

## 1. Stage Conclusion
**GO**

## 2. Exact Command
To run the fixed chair target sanity check script:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python verify_stageC6_fixed_chair_target_sanity.py
```
- **Target path**: `./targets/stage3_fixed_chair_target.npy`

## 3. Target Tensor Audit
- **Shape**: `[2048, 3]` (This will be correctly expanded to `[batch_size, 2048, 3]` during training/eval).
- **Dtype**: `float32`
- **Finite Ratio**: `1.0`
- **NaN / Inf count**: `0`
- **Mean / Std**: `-0.202665` / `0.557126`
- **Min / Max**: `-1.000000` / `1.000000`

## 4. Normalization Audit
- **bbox_min**: `[-0.8571, -0.6626, -1.0000]`
- **bbox_max**: `[0.8571, 0.6626, 1.0000]`
- **bbox_center**: `[0.0, 0.0, 0.0]`
- **bbox_extent**: `[1.7143, 1.3252, 2.0000]`
- **bbox_extent_max**: `2.000000`
- **max_abs_coord**: `1.000000`

**Conclusion**: The target point cloud is strictly and perfectly normalized according to the `shape_bbox` normalization rules (center exactly at 0, max extent exactly 2, max absolute coordinate exactly 1.0).

## 5. Visual Audit
- **Visualization path**: `summary_report/stageC/stageC6_fixed_chair_target.png`
- **Visual check**: Visually verified as chair-like from multiple angles (front, side, top).

## 6. Direction C Relevance
This verified fixed chair target acts as the *in-distribution* fixed target for the BadDiffusion-style backdoor fine-tuning (Stage C7). We are using an in-distribution (chair) target to eliminate OOD (Out-of-Distribution) difficulties that might obscure the mechanism validation. Using an airplane target is reserved for later stress-testing (Stage C11/C12) once the core Generation-Process Trigger pipeline is proven effective on the easier chair target.
