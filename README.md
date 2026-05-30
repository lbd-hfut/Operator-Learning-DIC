# Operator-Learning-DIC

基于 Neural Operator（神经算子）的数字图像相关（DIC）位移场测量框架。

核心思路：将 DIC 的灰度不变假设 `I_ref(x) = I_tar(x + u(x))` 形式化为算子学习问题，一次训练后可在任意分辨率、任意查询点上推理位移场。

## 1. 环境配置

```bash
conda create -n dic python=3.10
conda activate dic
pip install torch numpy scipy h5py matplotlib tqdm einops PyYAML pillow
```

## 2. 快速开始

```powershell
# 1. 准备散斑图放入 dataset/original_image/，编辑 config/dataset.yaml
# 2. 生成数据集
conda run -n dic python -m dataset.generate_dataset --config config/dataset.yaml

# 3. 训练
conda run -n dic python _train_simple.py --route A --steps 10000
conda run -n dic python _train_simple.py --route B --steps 5000 --batch_size 4

# 4. 预测 + 可视化
conda run -n dic python _predict.py --route A --ckpt checkpoints/route_a/best.pt --sample 0
conda run -n dic python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample 0
```

## 3. 数据准备

### 3.1 准备散斑原图

将真实散斑图案放入 `dataset/original_image/`（支持 png / jpg / bmp / tiff）。

```
dataset/original_image/
  speckle_001.png
  speckle_002.png
  ...
```

### 3.2 编辑数据集配置 `config/dataset.yaml`

```yaml
mode: real
real_image_dir: dataset/original_image
output_format: dir

image_size: [256, 256]

splits:
  train: 800
  validation: 100
  test: 100

deformation_modes:          # 权重控制各模式占比
  tension:             1.0   # 拉伸
  compression:         1.0   # 压缩
  shear:               1.0   # 剪切
  rotation:            1.0   # 旋转
  composite:           1.0   # 复合变形
  multiscale_random:   3.0   # 亚像素多尺度随机场（< 1 px）

displacement_range: [0.1, 20.0]   # 位移幅值范围 (px)
noise_std_range: [0.0, 0.03]      # 高斯噪声强度
seed: 42
```

### 3.3 生成数据集

```powershell
# 使用 YAML 配置生成
conda run -n dic python -m dataset.generate_dataset --config config/dataset.yaml

# 覆盖分集数量
conda run -n dic python -m dataset.generate_dataset --config config/dataset.yaml --train 100 --test 50
```

输出目录自动以时间戳命名：

```
dataset/dataset/2026-05-27/
  train/
    ref/         # 参考图 PNG（散斑原图裁剪）
    tar/         # 目标图 PNG（warp + 噪声后）
    u_field/     # 位移场 .npy [H, W, 2]
    roi_mask/    # ROI 掩码 PNG（255=有效像素）
    metadata.csv # index, deformation_mode, noise_std, roi_coverage
  validation/
    ...
  test/
    ...
```

### 3.4 CLI 参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | - | YAML 配置文件路径 |
| `--train / --val / --test` | 0 | 各分集样本数 |
| `--mode` | real | real / synthetic |
| `--real_image_dir` | dataset/original_image | 散斑图目录 |
| `--output_format` | dir | dir / h5 |
| `--output` | auto | 输出路径（默认时间戳） |
| `--image_size` | 256 256 | 图像尺寸 |
| `--seed` | 42 | 随机种子 |

## 4. 变形模式说明

| 模式 | 位移范围 | 公式 | 特点 |
|---|---|---|---|
| `tension` | 0.1 ~ 20 px | `u_x = A(x-0.5)`, `u_y = -0.3A(y-0.5)` | x 方向拉伸 + 泊松收缩 |
| `compression` | 0.1 ~ 20 px | 同上取负 | x 方向压缩 |
| `shear` | 0.1 ~ 20 px | `u_x = A(y-0.5)`, `u_y = 0` | 简单剪切 |
| `rotation` | 0.1 ~ 20 px | 绕中心旋转 | 刚体旋转 |
| `composite` | 0.1 ~ 20 px | 拉伸+剪切+正弦非线性 | 复合变形 |
| `multiscale_random` | 0.3 ~ 1.0 px | 多尺度随机控制点 + bicubic 插值 | 复杂亚像素场，边界置零 |

## 5. 训练

### 5.1 简易训练脚本 `_train_simple.py`

简洁训练脚本，支持 Route A 和 Route B，自动保存 best.pt / last.pt。

```powershell
# Route A（默认）
conda run -n dic python _train_simple.py

# Route A 自定义参数
conda run -n dic python _train_simple.py --route A --steps 10000 --lr 1e-4 --batch_size 8

# Route B（需减小 batch_size，GPU 显存 12G）
conda run -n dic python _train_simple.py --route B --steps 5000 --batch_size 4

# 指定数据集目录
conda run -n dic python _train_simple.py --route A --data_dir dataset/dataset/2026-05-27/train
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--route` | A | A 或 B |
| `--steps` | 10000 | 训练步数 |
| `--lr` | 1e-4 | 学习率 |
| `--batch_size` | 8 | 批大小（Route B 建议 4） |
| `--data_dir` | dataset/dataset/2026-05-27/train | 训练数据目录 |

输出：`checkpoints/route_a/{best.pt, last.pt}` 或 `checkpoints/route_b/{best.pt, last.pt}`

### 5.2 完整训练脚本 `train.py`

支持 YAML 配置、checkpoint 断点续训、DDP 多卡。

```powershell
# Route A — YAML 配置训练（推荐）
conda run -n dic python -m dic_solver_operator.train --config config/training.yaml

# Route A — 从 checkpoint 断点续训
conda run -n dic python -m dic_solver_operator.train --resume checkpoints/route_a/last.pt

# Route A — 文件夹数据集直接训练
conda run -n dic python -m dic_solver_operator.train --dataset_dir dataset/dataset/2026-05-27/train

# Route B 同理
conda run -n dic python -m deformation_inverse_operator.train --config config/training.yaml
conda run -n dic python -m deformation_inverse_operator.train --resume checkpoints/route_b/last.pt
conda run -n dic python -m deformation_inverse_operator.train --dataset_dir dataset/dataset/2026-05-27/train

# 多 GPU（DDP）
torchrun --nproc_per_node=4 -m dic_solver_operator.train --use_ddp --config config/training.yaml
torchrun --nproc_per_node=4 -m deformation_inverse_operator.train --use_ddp --config config/training.yaml
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | - | YAML 配置文件 |
| `--resume` | - | 从 checkpoint 续训 |
| `--dataset_dir` | - | 文件夹数据集路径（覆盖 YAML） |
| `--use_ddp` | False | 启用多卡分布式训练 |

## 6. 预测与可视化

### 6.1 命令行脚本 `_predict.py`

加载训练好的模型，在完整 256×256 网格上预测位移场，生成 3×3 对比图。

```powershell
# Route A — 预测单个样本
conda run -n dic python _predict.py --route A --ckpt checkpoints/route_a/best.pt --sample 0

# Route B — 预测并保存图片
conda run -n dic python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample 0 --save_plot checkpoints/route_b/pred_0.png

# 不传 --save_plot 时自动存到 predictions/route_X_000000.png
conda run -n dic python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample 5

# 批量预测（PowerShell 循环）
foreach ($s in 0,1,2,3,5,10) { conda run -n dic python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample $s }

# 预测全部 test 样本
foreach ($s in 0..99) { conda run -n dic python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample $s }
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--route` | A | A 或 B |
| `--ckpt` | - | checkpoint 路径（不存在时自动训练） |
| `--sample` | 0 | 样本编号 |
| `--data_dir` | dataset/dataset/2026-05-27/test | 数据目录 |
| `--save_plot` | auto | 图片保存路径（默认 `predictions/route_X_000000.png`） |

### 6.2 Python API `predict.py`

从代码中调用，适用于批量推理和集成。

```python
# Route A
import dic_solver_operator.predict as PA

# 一次性加载
u = PA.predict_dense(ref_img, tar_img, ckpt="checkpoints/route_a/best.pt")

# 复用 Predictor（避免重复加载）
pred = PA.Predictor("checkpoints/route_a/best.pt")
u_dense = pred.dense(ref_img, tar_img)          # [H, W, 2]
u_sparse = pred.sparse(ref_img, tar_img, pts)    # [N, 2]

# 编码一次、解码多次（高效批量查询）
f_enc = pred.encode(ref_img, tar_img)
u1 = pred.decode(points_a, f_enc)
u2 = pred.decode(points_b, f_enc)

# Route B — 接口完全一致
import deformation_inverse_operator.predict as PB

pred = PB.Predictor("checkpoints/route_b/best.pt")
u_dense = pred.dense(ref_img, tar_img)
u_sparse = pred.sparse(ref_img, tar_img, pts)
```

输入：
- `ref_img` / `tar_img` — `np.ndarray [H, W]`，值域 `[0, 1]`（float32）
- `query_points` — `np.ndarray [N, 2]`，归一化坐标 `[0, 1]²`（x, y 顺序）

返回：
- `u` — `np.ndarray [H, W, 2]` 或 `[N, 2]`，位移单位为**像素**

## 7. 不规则 ROI 测试 `_test_irregular_roi.py`

将参考图 ROI 外区域置黑，用 GT 位移场 warp 生成变形图，验证模型在不规则 ROI 下的预测能力。

```powershell
# 圆形 ROI
conda run -n dic python _test_irregular_roi.py --sample 0 --roi_type circle

# 环形 ROI
conda run -n dic python _test_irregular_roi.py --sample 0 --roi_type ring

# 缺口板 (dogbone + 中心孔)
conda run -n dic python _test_irregular_roi.py --sample 0 --roi_type notch

# 使用数据集自带 ROI
conda run -n dic python _test_irregular_roi.py --sample 0

# 指定输出路径
conda run -n dic python _test_irregular_roi.py --sample 5 --roi_type circle --save_plot predictions/roi_sample5.png
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sample` | 0 | 样本编号 |
| `--roi_type` | None | circle / ellipse / ring / notch（None=使用数据集ROI） |
| `--data_dir` | dataset/dataset/2026-05-27/test | 数据目录 |
| `--ckpt_a` | checkpoints/route_a/best.pt | Route A checkpoint |
| `--ckpt_b` | checkpoints/route_b/best.pt | Route B checkpoint |
| `--save_plot` | predictions/irregular_roi.png | 结果图片路径 |

## 8. 架构说明

### Route A：DIC Solver Operator

```
Ref + Tar + Diff (3ch)
       │
  DualChannelCNN
  (stride-2 in first stage)
       │
  F_input [B, 16384, 256]
       │
  SimpleLocalDecoder
  ┌─────────────────────┐
  │ bilinear_sample(F, q) → f_local
  │ GFF(q) → pos
  │ MLP([pos, f_local]) → u(q)
  └─────────────────────┘
```

编码器在第一个卷积层直接比较 ref/tar/diff，局部特征采样解码。

### Route B：Deformation Inverse Operator

```
Ref (1ch)          Tar (1ch)
    │                  │
SiameseCNN (shared)  SiameseCNN (shared)
    │                  │
F_ref [B, N, 256]   F_tar [B, N, 256]
    └──────┬───────────┘
           │
  InverseDecoder
  ┌─────────────────────────────────┐
  │ bilinear_sample(F_ref, q) → f_ref_local
  │ bilinear_sample(F_tar, q) → f_tar_local
  │ f_diff = f_tar - f_ref
  │ GFF(q) → pos
  │ MLP([pos, f_ref, f_tar, f_diff]) → u(q)
  └─────────────────────────────────┘
```

孪生 CNN 分别编码参考图和目标图，解码器在查询点同时采样两个特征图并通过特征差推断位移。

### 设计要点

- **无交叉注意力**：Two 路线均使用双线性特征采样 + MLP 解码，避免 Galerkin 交叉注意力的 K^T@V 瓶颈导致的模式坍塌
- **GFF 位置编码**：高斯傅里叶特征编码查询坐标，使 MLP 能表示高频位移变化
- **局部特征采样**：`grid_sample` 在每个查询点提取局部编码器特征，而非全局聚合

## 9. 项目结构

```
├── common/                         # 共享组件
│   ├── cross_attention.py          # Galerkin 线性交叉注意力
│   ├── self_attention.py           # 线性自注意力
│   ├── gaussian_fourier_features.py # GFF 坐标编码
│   ├── feedforward.py              # FFN
│   ├── layer_norm.py               # LayerNorm / PostNorm
│   ├── losses.py                   # 复合损失函数
│   ├── checkpoint.py               # checkpoint 存取
│   └── config_utils.py             # YAML 配置加载
├── dataset/                        # 数据管线
│   ├── image_pool.py               # 真实散斑图加载
│   ├── deformation_generator.py    # 位移场生成（6 种模式）
│   ├── warp.py                     # 正映射 splatting
│   ├── roi.py                      # ROI 计算
│   ├── generate_dataset.py         # 离线数据集生成 CLI
│   ├── folder_dataset.py           # 训练时读取文件夹数据集
│   ├── dic_dataset.py              # 在线合成数据集
│   ├── collate.py                  # batch 整理
│   └── sampler.py                  # 查询点采样
├── dic_solver_operator/            # Route A
│   ├── config.py                   # 模型配置
│   ├── encoder.py                  # DualChannelCNN
│   ├── decoder.py                  # SimpleLocalDecoder
│   ├── model.py                    # SolverOperatorModel
│   ├── predict.py                  # 预测 API
│   └── train.py                    # 完整训练脚本
├── deformation_inverse_operator/   # Route B
│   ├── config.py                   # 模型配置
│   ├── encoder.py                  # SiameseCNN + DiffCrossAttnEncoder
│   ├── decoder.py                  # InverseDecoder
│   ├── model.py                    # InverseOperatorModel
│   ├── predict.py                  # 预测 API
│   └── train.py                    # 完整训练脚本
├── config/
│   ├── dataset.yaml                # 数据集生成配置
│   └── training.yaml               # 训练配置
├── checkpoints/
│   ├── route_a/                    # Route A 模型权重
│   └── route_b/                    # Route B 模型权重
├── predictions/                    # 预测结果图片
├── _train_simple.py                # 简易训练脚本
├── _predict.py                     # 预测 + 可视化脚本
├── _test_irregular_roi.py          # 不规则 ROI 测试脚本
└── experiments/                    # 实验配置和启动脚本
```

## 10. 技术要点

- **Galerkin 线性注意力**：无 softmax，O(N·d²) 复杂度，K 列 InstanceNorm 保证基函数单位范数
- **正映射 splatting**：参考图像素按位移推至目标位置 + 双线性权重分散
- **ROI 自动计算**：逐像素判定 `x + u(x)` 是否在图像边界内
- **模式坍塌修复**：移除交叉注意力中的 V InstanceNorm + 最终替换为局部特征采样解码器
