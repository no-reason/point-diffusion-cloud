import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from torch.utils import data

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dgcnn import DGCNN 
from datasets import ShapeNetPartDataset

def lr_lambda(step):
    if step < 10:
        return 1
    elif 10 <= step < 30:
        return 0.1
    else:
        return 0.01

class ClassAwareBase:
    """
    提供 ClassAwareRotation / ClassAwarePerturb 共享的工具方法
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_cat2id(self, dataset_root: str, class_choice: list = None) -> dict:
        """
        读取 synsetoffset2category.txt 并根据 class_choice 过滤。
        """
        cat2id = {}
        synset_file = os.path.join(dataset_root, 'synsetoffset2category.txt')
        # 直接 open（若不存在则抛错）
        with open(synset_file) as f:
            for ln in f:
                cname, cid = ln.strip().split()
                cat2id[cname] = cid
        if class_choice is not None:
            cat2id = {k: v for k, v in cat2id.items() if k in class_choice}
        return cat2id

    def _build_class_maps(self, cat2id: dict):
        """
        根据 cat2id 构建 classes_map 和 id2cat
        """
        classes_map = dict(zip(range(len(cat2id)), sorted(cat2id)))
        id2cat = {i: name for i, name in classes_map.items()}
        return classes_map, id2cat

    def _ensure_cache_dir(self, dataset_root: str, cache_dir: str, default_subdir: str) -> str:
        if cache_dir is None:
            cache_dir = os.path.join(dataset_root, default_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _get_model(self, ckpt_path):        
        latent_dim = 512
        encoder = DGCNN(emb_dims=latent_dim).to("cpu")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")
        try:
            full_checkpoint = torch.load(ckpt_path, map_location="cpu")
            if 'model_state_dict' in full_checkpoint:
                full_state_dict = full_checkpoint['model_state_dict']
            else:
                full_state_dict = full_checkpoint
            encoder_state_dict = {}
            for key, value in full_state_dict.items():
                if key.startswith('encoder.'):
                    new_key = key[len('encoder.'):]
                    encoder_state_dict[new_key] = value
            
            if not encoder_state_dict:
                raise KeyError("Checkpoint does not contain any keys with 'encoder.' prefix.")

            encoder.load_state_dict(encoder_state_dict)
            print(f"Successfully extracted and loaded DGCNN encoder weights from {ckpt_path}")

        except Exception as e:
            print(f"Warning: Failed to load encoder weights from {ckpt_path}. Using random weights. Error: {e}")
        encoder.eval()
        return encoder

    def _build_ref_features_library(
        self,
        dataset_root: str,
        ref_split: str,
        batch_size: int,
        cache_dir: str
    ) -> dict:
        ref_features_library = {}
        for class_name in self.cat2id.keys():
            cache_file = os.path.join(cache_dir, f"{class_name.replace(' ', '_')}.pt")
            if os.path.exists(cache_file):
                print(f"  - 从 {cache_file} 加载类别 '{class_name}' 的缓存特征")
                ref_features_library[class_name] = torch.load(cache_file, map_location='cpu')
            else:
                print(f"  - 为类别 '{class_name}' 构建特征 (未找到缓存)...")
                features = self._build_single_class_ref_features(
                    dataset_root, class_name, ref_split, batch_size
                )
                ref_features_library[class_name] = features
                print(f"  - 缓存特征到 {cache_file}")
                torch.save(features, cache_file)
        print("Reference feature library built and cached for all classes.")
        return ref_features_library

    def _build_single_class_ref_features(
        self,
        dataset_root: str,
        class_name: str,
        split: str,
        batch_size: int
    ) -> torch.Tensor:
        """为单个指定类别构建特征参考库"""
        dataset = ShapeNetPartDataset(
            root=dataset_root,
            split=split,
            npoints=2048,
            classification=True,
            data_augmentation=False,
            class_choice=[class_name]
        )
        if len(dataset) == 0:
            raise ValueError(f"Class '{class_name}' not found in the dataset or the subset is empty.")
            
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        class_features = []
        with torch.no_grad():
            for points, _ in tqdm(dataloader, desc=f"  Extracting features for {class_name}"):
                points = points.to(self.device)
                features, _ = self.model(points)
                class_features.append(features.cpu())
        
        return torch.cat(class_features, dim=0)

    def _feature_extractor(self, pc: torch.Tensor, req_grad: bool = False) -> torch.Tensor:
        if pc.dim() == 2: pc = pc.unsqueeze(0)
        if req_grad: feat, _ = self.model(pc)
        else:
            with torch.no_grad(): feat, _ = self.model(pc)
        return feat

    def _delete_x_feature(self, z_current: torch.Tensor, Z_refs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_current_2d = z_current.squeeze(0) if z_current.dim() > 1 else z_current
            norms = torch.norm(Z_refs - z_current_2d, p=2, dim=1)
            closest_idx = torch.argmin(norms).item()
            mask = torch.ones(Z_refs.shape[0], dtype=torch.bool, device=Z_refs.device)
            mask[closest_idx] = False
            Zx = Z_refs[mask]
        return Zx


class ClassAwareRotation(ClassAwareBase):
    """
    一个动态的、针对任意类别的后门触发器生成器。
    在初始化时，它会为数据集中所有指定的类别构建一个特征参考库。
    在调用时，根据传入的类别标签，选择对应的特征库进行优化。
    """
    def __init__(self,
                 pretrained_ckpt_path: str,
                 dataset_root: str,
                 class_choice: list = None,  # 可以指定要为哪些类构建特征库，默认为全部
                 cache_dir: str = None,
                 # 优化参数
                 iters: int = 10,
                 lr: float = 0.1,
                 # 特征集构建参数
                 ref_split='train',
                 batch_size=32):
        super().__init__()
        self.iters = iters
        self.lr = lr

        self.cat2id = self._load_cat2id(dataset_root=dataset_root, class_choice=class_choice)
        self.classes_map, self.id2cat = self._build_class_maps(self.cat2id)
        self.model = self._get_model(pretrained_ckpt_path).to(self.device)
        cache_dir = self._ensure_cache_dir(dataset_root, cache_dir, 'optClassAware_feature_cache')
        self.ref_features_library = self._build_ref_features_library(
            dataset_root=dataset_root,
            ref_split=ref_split,
            batch_size=batch_size,
            cache_dir=cache_dir
        )

    def _euler_to_rotation_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        根据欧拉角计算旋转矩阵，并保持计算图连接以进行反向传播。
        """
        c1, s1 = torch.cos(theta[0]), torch.sin(theta[0])
        c2, s2 = torch.cos(theta[1]), torch.sin(theta[1])
        c3, s3 = torch.cos(theta[2]), torch.sin(theta[2])
        zero = torch.tensor(0.0, device=self.device)
        one = torch.tensor(1.0, device=self.device)
        Rx = torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, c1, -s1]),
            torch.stack([zero, s1, c1])
        ])
        Ry = torch.stack([
            torch.stack([c2, zero, s2]),
            torch.stack([zero, one, zero]),
            torch.stack([-s2, zero, c2])
        ])
        Rz = torch.stack([
            torch.stack([c3, -s3, zero]),
            torch.stack([s3, c3, zero]),
            torch.stack([zero, zero, one])
        ])
        return Rz @ Ry @ Rx

    def __call__(self, pos_np: np.ndarray, pc_cls: int):
        # 根据传入的类别标签，动态选择特征库
        ref_class_name = self.id2cat.get(pc_cls)
        if ref_class_name is None:
            raise ValueError(f"Received invalid class index: {pc_cls}")
        if ref_class_name not in self.ref_features_library:
            raise ValueError(f"Trigger was not initialized for class '{ref_class_name}'.")
        
        ref_features = self.ref_features_library[ref_class_name].to(self.device)

        pos_orig = pos_np.astype(np.float32).copy()

        centroid = np.mean(pos_orig, axis=0)
        pos_centered = pos_orig - centroid
        scale = np.max(np.linalg.norm(pos_centered, axis=1))
        if scale == 0: scale = 1.0  # 防止除以0
        pos_norm_np = pos_centered / scale
        
        pos_tensor = torch.tensor(pos_norm_np, dtype=torch.float32, device=self.device)

        z_current = self._feature_extractor(pos_tensor).detach()
        # 注意：这里的 z_current 和 ref_features 维度可能不完全匹配，需要统一
        z_current = z_current.squeeze(1) if z_current.dim() == 3 else z_current # (1, D) -> (D)
        ref_features_squeezed = ref_features.squeeze(1) if ref_features.dim() == 3 else ref_features

        Zx = self._delete_x_feature(z_current, ref_features_squeezed) # (M-1, d)

        with torch.enable_grad():     
            euler_angles = nn.Parameter(torch.zeros(3, device=self.device))
            optimizer = torch.optim.Adam([euler_angles], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            center = pos_tensor.mean(dim=0, keepdim=True)
        
            for i in range(self.iters):
                optimizer.zero_grad()
                R = self._euler_to_rotation_matrix(euler_angles)
                pos_rotated = (pos_tensor - center) @ R.T + center
                z_adv = self._feature_extractor(pos_rotated, req_grad=True)
                loss = F.cosine_similarity(z_adv.squeeze().expand_as(Zx), Zx, dim=1).mean()
                loss.backward()
                # if i % 10 == 0: # 每10次迭代打印一次
                #     print(f"Iter {i}, Loss: {loss.item():.4f}, Grad: {euler_angles.grad.detach().cpu().numpy()}")
                optimizer.step()
                scheduler.step()

        final_angles_rad = euler_angles.detach()
        final_R = self._euler_to_rotation_matrix(final_angles_rad)
        pos_rotated_norm = ((pos_tensor - center) @ final_R.T + center).cpu().numpy()
        final_rotated_pos_np = pos_rotated_norm * scale + centroid
        final_angles_deg = np.rad2deg(final_angles_rad.cpu().numpy())
        
        return pos_orig, final_rotated_pos_np, final_angles_deg



class ClassAwarePerturb(ClassAwareBase):
    """
    一个动态的、针对任意类别的后门触发器生成器。
    通过优化一个小的加性扰动 delta，使得点云的特征在同类中尽可能疏远。
    最后再添加一个固定的球形触发器。
    """
    def __init__(self,
                 pretrained_ckpt_path: str,
                 dataset_root: str,
                 class_choice: list = None,
                 cache_dir: str = None,
                 # 优化参数
                 iters: int = 20, # 一般20次迭代足够
                 lr: float = 0.01,
                 perturb_bound_C: float = 0.04, # L_inf 约束
                 # 特征集构建参数
                 ref_split='train',
                 batch_size=32):
        super().__init__()
        self.iters = iters
        self.lr = lr
        self.perturb_bound_C = perturb_bound_C

        self.cat2id = self._load_cat2id(dataset_root=dataset_root, class_choice=class_choice)
        self.classes_map, self.id2cat = self._build_class_maps(self.cat2id)
        self.model = self._get_model(pretrained_ckpt_path).to(self.device)
        cache_dir = self._ensure_cache_dir(dataset_root, cache_dir, 'optClassAware_feature_cache')
        self.ref_features_library = self._build_ref_features_library(
            dataset_root=dataset_root,
            ref_split=ref_split,
            batch_size=batch_size,
            cache_dir=cache_dir
        )

    def __call__(self, pos_np: np.ndarray, pc_cls: int):
        ref_class_name = self.id2cat.get(pc_cls)
        if ref_class_name is None: raise ValueError(f"Invalid class index: {pc_cls}")
        if ref_class_name not in self.ref_features_library: raise ValueError(f"Trigger not initialized for class '{ref_class_name}'.")
        
        ref_features = self.ref_features_library[ref_class_name].to(self.device)
        pos_orig = pos_np.astype(np.float32).copy()

        centroid = np.mean(pos_orig, axis=0)
        pos_centered = pos_orig - centroid
        scale = np.max(np.linalg.norm(pos_centered, axis=1))
        if scale == 0: scale = 1.0  # 防止除以0
        pos_norm_np = pos_centered / scale
        pos_tensor = torch.tensor(pos_norm_np, dtype=torch.float32, device=self.device)

        z_current = self._feature_extractor(pos_tensor).detach()
        z_current = z_current.squeeze(1) if z_current.dim() == 3 else z_current
        ref_features_squeezed = ref_features.squeeze(1) if ref_features.dim() == 3 else ref_features
        Zx = self._delete_x_feature(z_current, ref_features_squeezed)

        with torch.enable_grad():     
            delta = nn.Parameter(torch.zeros_like(pos_tensor))
            optimizer = torch.optim.Adam([delta], lr=self.lr)
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) # 可选
        
            for i in range(self.iters):
                optimizer.zero_grad()
                pos_perturbed = pos_tensor + delta
                z_adv = self._feature_extractor(pos_perturbed, req_grad=True)
                loss = F.cosine_similarity(z_adv.squeeze().expand_as(Zx), Zx, dim=1).mean()                
                loss.backward()
                # if i % 10 == 0:
                #     print('Loss: ', loss)
                optimizer.step()
                # scheduler.step() # 可选
                # 投影到 L_infinity 范数球内
                with torch.no_grad():
                    delta.data.clamp_(-self.perturb_bound_C, self.perturb_bound_C)
            
        perturbed_pos_np = (pos_tensor + delta.detach()).cpu().numpy()
        poison_pos = perturbed_pos_np * scale + centroid

        return pos_orig, poison_pos