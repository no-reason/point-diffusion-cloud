import os
import numpy as np

# ===== 路径自己改一下 =====
save_dir = '/data/personal_data/yzf/diffusion-point-cloud/results/AE_Ours_all_1767250238'
ref_npy = os.path.join(save_dir, 'ref.npy')
out_npy = os.path.join(save_dir, 'out.npy')

# 读取 npy
ref = np.load(ref_npy)   # 形状大概是 [N, P, 3]
out = np.load(out_npy)   # 同上

print('ref shape:', ref.shape)
print('out shape:', out.shape)

# 创建保存 .pts 的文件夹
ref_dir = os.path.join(save_dir, 'ref_pts')
out_dir = os.path.join(save_dir, 'recon_pts')
os.makedirs(ref_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

num_samples = ref.shape[0]

for i in range(num_samples):
    ref_i = ref[i]   # [P, 3]
    out_i = out[i]   # [P, 3]

    ref_path = os.path.join(ref_dir, f'ref_{i:05d}.pts')
    out_path = os.path.join(out_dir, f'recon_{i:05d}.pts')

    # 不带表头版本：每行 x y z
    np.savetxt(ref_path, ref_i, fmt='%.6f')
    np.savetxt(out_path, out_i, fmt='%.6f')

print(f'Saved {num_samples} ref pts to {ref_dir}')
print(f'Saved {num_samples} recon pts to {out_dir}')
