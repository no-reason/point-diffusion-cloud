import torch
import numpy as np
import os
# 导入您现有的 Dataset 类
from utils.dataset import ShapeNetCore

# 1. 配置路径 (请确保这里指向您的 .h5 文件路径)
DATASET_PATH = './data/shapenet_v2pc15k.h5' 

# 2. 指定我们要提取的类别：'earphone' (耳机)
TARGET_CATE = ['earphone'] 

print(f"正在从 {DATASET_PATH} 中提取【耳机】样本...")

# 3. 加载数据集 (只加载耳机这一类)
# 注意：scale_mode='shape_unit' 保证提取出的耳机是归一化好的，适合做目标
dset = ShapeNetCore(path=DATASET_PATH, cates=TARGET_CATE, split='train', scale_mode='shape_unit')

if len(dset) == 0:
    print("错误：没有找到耳机数据！请检查路径或 dataset.py 中的字典。")
else:
    # 4. 挑选一个样本 (这里选第 0 个，您也可以随机选或者挑一个好看的)
    # 建议多运行几次或者 print 一下 index，确保选出来的形状比较标准
    pick_index = 0
    data = dset[pick_index]
    target_pc = data['pointcloud'].numpy()
    
    # 5. 保存为 target_earphone.npy
    save_name = 'target_earphone.npy'
    np.save(save_name, target_pc)
    print(f"提取成功！")
    print(f"已保存目标文件: {save_name}")
    print(f"点云形状: {target_pc.shape}")
    print(f"该样本原始类别: {data['cate']}")