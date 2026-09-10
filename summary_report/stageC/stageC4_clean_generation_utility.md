# Stage C4: Formal Clean Generation Utility Verification

## 1. Stage Conclusion
**GO**

## 2. Exact Command
To run the clean generation utility baseline script:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python verify_stageC4_clean_generation_utility.py
```
- **Checkpoint path**: `./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Number of generated samples**: `128`
- **Reference dataset path**: `./data/shapenet_v2pc15k.h5` (ShapeNetCore, split='test', cates=['chair'])

## 3. Sample Statistics
- **Generated sample shape**: `[128, 2048, 3]`
- **Dtype**: `torch.float32`
- **Min / Max**: `-3.5274` / `3.2200`
- **Mean / Std**: `0.0015` / `0.9438`
- **NaN / Inf count**: `0`
- **Finite ratio**: `1.0`

## 4. Generation Metrics (Clean Baseline)
Computed using standard L2 squared Chamfer Distance (bidirectional mean sum, unscaled by 0.5) against a random 128-sample reference subset from the test split.

- **MMD-CD**: `0.5302`
- **COV-CD**: `0.1015` (10.15% coverage over the 128 reference samples)
- **1NN-CD**: `1.0000` 

*Note regarding 1NN-CD: 1NN-CD measures the distinguishability between the generated and reference distributions using a 1-Nearest-Neighbor classifier. A score of 1.0 (100%) indicates that the distributions are perfectly separable in the Chamfer Distance space under this small sample size (128 vs 128). An ideal score is around 0.50 (50%), where generated samples are indistinguishable from real samples. The current score (1.0) and coverage (0.10) reflect the generative capability of this specific checkpoint at N=128, which serves as our anchor baseline for future post-backdoor utility evaluations.*

## 5. Visualizations & Outputs
- **Visualization path**: `summary_report/stageC/stageC4_clean_generation_vis.png`
- **Samples path**: `summary_report/stageC/stageC4_clean_generation_samples.npz`
- **Metrics JSON**: `summary_report/stageC/stageC4_clean_generation_metrics.json`
- **Log**: `summary_report/stageC/stageC4_clean_generation_smoke.log`

## 6. Clean Generation Utility Conclusion
The generated samples are perfectly finite with zero NaNs or Infs, and the metrics are successfully computed using the native `compute_all_metrics_lion` method (without Ninja C++ extension errors, by isolating the CD-only path). The generation does not break down and establishes a solid numerical baseline against which we can compare the backdoored model in Stage C10. No optimizer steps or checkpoint modifications occurred. 
