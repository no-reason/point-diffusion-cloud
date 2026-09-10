# Stage C0: Clean Generation Pipeline Audit

## 1. Stage Conclusion
**GO**

## 2. Exact Command
To run the clean generation smoke test:
```bash
/root/anaconda3/envs/baddiffusion-img/bin/python test_gen_smoke.py
```

## 3. Checkpoint Path
`./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt` (Trained on chair category only).

## 4. Sample Entry Function
The generation entry function is `GaussianVAE.sample(self, z, num_points, flexibility, truncate_std=None)` in `models/vae_gaussian.py`.
It takes a randomly sampled normal latent `z` as the shape code and forwards it to the diffusion model.

## 5. Reverse Diffusion Function
The reverse diffusion function is `DiffusionPoint.sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False)` in `models/diffusion.py`.
It starts by initializing `x_T = torch.randn([batch_size, num_points, point_dim])`, then iteratively denoises it using the `PointwiseNet` given the `context` (which is `z`).

## 6. Whether Encoder Input is Used
**No**. The encoder is strictly used during the training phase (`get_loss`) to map a ground truth point cloud to a latent `z`. During sampling/generation, `z` is sampled directly from a standard normal distribution `N(0, I)`, and `x_T` is also sampled from `N(0, I)`. No input point cloud is fed to the model during the pure generation process.

## 7. Tensor Shapes and Devices
During the clean generation smoke test (batch size = 8, sample points = 2048):
- **z**: `torch.Size([8, 512])`, dtype: `torch.float32`, device: `cuda:0`
- **X_T**: `torch.Size([8, 2048, 3])`, dtype: `torch.float32`, device: `cuda:0`
- **X_0** (samples): `torch.Size([8, 2048, 3])`, dtype: `torch.float32`, device: `cuda:0`

## 8. Finite Ratio
**1.0** (No NaN/Inf detected in the generated point clouds).

## 9. Match with Direction C
This pipeline **perfectly matches Direction C**. The model is purely generative at inference. The process `z ~ N(0, I) -> X_T ~ N(0, I) -> reverse diffusion -> X_0` is fully isolated from any semantic source point cloud input. This creates an ideal attack surface for injecting a backdoor directly into the generation setting (e.g., modifying `X_T` or `X_t`).

