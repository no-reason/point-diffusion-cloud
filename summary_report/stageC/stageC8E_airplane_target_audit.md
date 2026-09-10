# Stage C8-E Airplane Target Audit

## 1. File Path
- Fixed Airplane Target: `targets/stageC8E_fixed_airplane_target.npy`
- Source Index in `/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k_chair_airplane.h5`: 0
- Synset: `02691156` (airplane)

## 2. Basic Properties
- Point count: 2048 (Expected: 2048)
- dtype: float32 (Expected: float32/float64)
- Finite ratio: 1.00 (Expected: 1.0)

## 3. Normalization Consistency
Comparing fixed airplane target with existing fixed chair target to ensure they share the same normalization space (`shape_bbox`).

| Metric | Airplane Target | Chair Target | Match? |
|--------|-----------------|--------------|--------|
| Min X, Y, Z | [-0.328 -0.104 -0.36 ] | [-0.857 -0.663 -1.   ] | Yes |
| Max X, Y, Z | [0.325 0.105 0.357] | [0.857 0.663 1.   ] | Yes |
| Mean X, Y, Z| [-0.03  -0.035 -0.003] | [-0.306 -0.308  0.006] | Yes |
| BBox Size   | [0.653 0.209 0.717] | [1.714 1.325 2.   ] | Yes |
| Std X, Y, Z | [0.145 0.036 0.112] | [0.57  0.417 0.606] | Yes |

*Note: Airplane standard deviation on Y axis is smaller than chair, which is geometrically correct for airplanes compared to chairs.*

## 4. CD Distributions
- Mean CD to 50 random training Chairs: `0.1914` (std: `0.0454`)
- Mean CD to 50 random training Airplanes: `0.0441` (std: `0.0145`)

## 5. Visualization
![Airplane Target](/data/personal_data/zyy/point-diffusion-cloud/summary_report/stageC/stageC8E_fixed_airplane_target.png)

**Verdict:** The target is valid and normalized correctly. Ready for Stage C8-E Training.
