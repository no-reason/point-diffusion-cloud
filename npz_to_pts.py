import numpy as np
import os

# ================= 配置 =================
# 输入文件路径
INPUT_PATH = '/data/personal_data/yzf/diffusion-point-cloud/target_earphone.npy'
# 输出文件路径 (自动替换后缀)
OUTPUT_PATH = INPUT_PATH.replace('.npy', '.pts')

def convert_npy_to_pts():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到文件 {INPUT_PATH}")
        return

    print(f"📂 正在加载: {INPUT_PATH}")
    # 1. 加载数据
    data = np.load(INPUT_PATH)
    print(f"ℹ️ 原始数据形状: {data.shape}")

    # 2. 处理维度
    # 如果是 (1, N, 3) 或 (B, N, 3)，只取第一个样本
    if len(data.shape) == 3:
        print("⚠️ 检测到 3D 数组，默认取第一个样本 (index 0)...")
        points = data[0]
    elif len(data.shape) == 2:
        points = data
    else:
        print(f"❌ 错误: 数据形状不正确，应该是 (N, 3) 或 (1, N, 3)，当前是 {data.shape}")
        return

    # 3. 归一化检查 (可选，方便查看，不改变原始数据)
    # print(f"   Max: {points.max():.4f}, Min: {points.min():.4f}")

    # 4. 保存为 .pts
    # 格式：X Y Z (空格分隔，保留6位小数)
    try:
        np.savetxt(OUTPUT_PATH, points, fmt='%.6f', delimiter=' ')
        print(f"✅ 转换成功！")
        print(f"💾 已保存至: {OUTPUT_PATH}")
        print(f"   包含点数: {points.shape[0]}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

if __name__ == "__main__":
    convert_npy_to_pts()