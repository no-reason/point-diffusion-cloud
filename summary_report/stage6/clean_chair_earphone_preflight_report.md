# Stage 6 Preflight Report: Clean Chair+Earphone VAE Training

## 1. 为什么需要训练新的 chair+earphone clean checkpoint

后续任务目标是进行 `chair -> earphone` 的 backdoor 攻击实验（即在 clean sample 是 chair 的情况下，注入 trigger 后使其生成 earphone）。由于当前的 Clean VAE checkpoint 仅在 `chair` 类别上进行了预训练（chair-only），模型在其 latent space 及 decoder 中从未见过 `earphone` 的分布数据。因此，如果在后续 backdoor finetuning 阶段，期望模型能够成功重构出复杂的 earphone，需要极高的代价，而且可能无法收敛（模型缺乏 earphone 形状的先验分布知识）。为了避免由于“未见过 earphone 数据”导致的分布外（OOD）生成灾难，我们需要训练一个新的包含 chair 和 earphone 两种类别数据的 Clean VAE checkpoint 作为 baseline/finetune 起点。

## 2. 当前 chair-only checkpoint 的局限

当前的 baseline (如 `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`) 完全是一个 chair-only 的模型。当输入 `earphone` 对应的 point cloud 时，模型不仅无法在 latent 空间提供良好的特征表达，其 decoder 也无法生成耳机形状（仅能解码成椅子相关的形状）。这种局限性使得其不能被直接用作 `chair -> earphone` target 攻击的起始模型。直接在 chair-only 模型上进行 backdoor 训练，会在一定程度上造成明显的分布偏置，使得 attack 难以成功。

## 3. 数据集中 chair 是否存在

**是。** 
根据数据集中的索引，类别名 `chair` (synset ID `03001627`) 是存在的。
当前 H5 训练集中，共提取到 **3746** 个 `chair` 样本。

## 4. 数据集中 earphone 是否存在

**是。**
根据数据集中的索引，类别名 `earphone` (synset ID `03261776`) 是存在的。
当前 H5 训练集中，共提取到 **69** 个 `earphone` 样本。这虽然数量较少（相较于 chair），但足以供模型在预训练时学习其分布模式。

## 5. chair sample stats

在 `ShapeNetCore` pipeline (`scale_mode='shape_bbox'`) 加载后：
- `finite_ratio`: 1.0000
- `min`: -1.0000
- `max`: 1.0000
- `max_abs`: 1.0000
- `bbox_center`: 0.0000 (近似为0)
- `bbox_extent_max`: 2.0000

## 6. earphone sample stats

在 `ShapeNetCore` pipeline (`scale_mode='shape_bbox'`) 加载后：
- `finite_ratio`: 1.0000
- `min`: -1.0000
- `max`: 1.0000
- `max_abs`: 1.0000
- `bbox_center`: 0.0000 (近似为0)
- `bbox_extent_max`: 2.0000

## 7. normalization 是否通过

**通过。** 
`chair` 和 `earphone` 数据在使用 `shape_bbox` 模式后均完美符合以下检查标准：
- `finite_ratio = 1.0`
- `max_abs <= 1.05` (当前值为 1.0000)
- `bbox_center 近似 0` (当前值为 0.0000)
- `bbox_extent_max 近似 2.0` (当前值为 2.0000)

## 8. clean training 脚本是否支持多类别

**支持。**
`train_gen.py` 的 argparse `categories` 参数默认接收列表，且 `ShapeNetCore` 本身便支持通过传入列表形式的 `cates`（例如 `['chair', 'earphone']`）在单一循环中并行加载多个类别的点云。因此现有训练入口完全支持多类别的无缝联合训练。

## 9. 推荐的新 checkpoint 输出目录

为了避免覆盖旧的 chair-only checkpoint，请使用包含 `Chair_Earphone` 以及日期标记的新文件夹目录。
- **推荐输出 tag：** `Clean_VAE_Chair_Earphone_KL001`
- **实际生成路径将形如：** `logs_gen/GEN_YYYY_MM_DD_Clean_VAE_Chair_Earphone_KL001/`
*(old checkpoint = chair-only, new checkpoint = chair+earphone clean model candidate)*

## 10. 推荐训练命令

（**请勿在此阶段执行**）
```bash
python train_gen.py \
    --tag Clean_VAE_Chair_Earphone_KL001 \
    --model gaussian \
    --categories chair earphone \
    --dataset_path ./data/shapenet_v2pc15k.h5 \
    --kl_weight 0.001 \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --scale_mode shape_unit \
    --normalize shape_bbox
```

*注：在原始 chair-only checkpoint 训练中使用了 `--scale_mode shape_unit` 来规范化训练数据，并在 test 时使用了 `--normalize shape_bbox`。建议保持一致，或者根据策略显式调整。*

## 11. 是否允许进入 clean chair+earphone training

**Verdict: GO**

理由：
1. `chair` 和 `earphone` 数据都存在，且样本量及形状维度 (`[B, 2048, 3]`) 读取正常。
2. 数据均满足 `shape_bbox` normalization。
3. `train_gen.py` 脚本原生支持多类别 list。
4. 提供不同的 `tag` 参数不会覆盖原有 `logs_gen` 下的 checkpont。
