# Operator-Learning-DIC

基于 Neural Operator（神经算子）与深度学习的数字图像相关（DIC）位移场测量框架。

核心思路：将 DIC 的灰度不变假设 `I_ref(x) = I_tar(x + u(x))` 形式化为算子学习问题，一次训练后可在任意分辨率、任意查询点上推理位移场。

提供四条路线：算子学习（Route A/B）、Transformer DIC（Route D）与传统 U-Net 密集预测（Route C）。

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

# 3. 训练（四选一）
conda run -n dic python _train_simple.py --route A --steps 10000
conda run -n dic python _train_simple.py --route B --steps 5000 --batch_size 4
conda run -n dic python _train_simple.py --route C --steps 5000 --batch_size 4
conda run -n dic python _train_simple.py --route D --steps 5000 --batch_size 4

# 4. 预测 + 可视化
conda run -n dic python _predict.py --route A --ckpt checkpoints/route_a/best.pt --sample 0
conda run -n dic python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample 0
conda run -n dic python _predict.py --route C --ckpt checkpoints/route_c/best.pt --sample 0
conda run -n dic python _predict.py --route D --ckpt checkpoints/route_d/best.pt --sample 0
```

## 3. 四条路线对比

| | Route A | Route B | Route C | Route D |
|---|---|---|---|---|
| **方法** | 双通道 CNN + 查询点解码 | 孪生 CNN + 查询点解码 | U-Net 密集预测 | CNN+ViT Transformer |
| **输入** | [ref, tar, diff] 3ch | ref 1ch + tar 1ch (独立) | [ref, tar] 2ch | ref 1ch + tar 1ch |
| **推理方式** | 逐查询点 | 逐查询点 | 一次前向全图 | 逐 token（分块解码） |
| **参数量** | ~23.6M | ~23.6M | ~31.0M | ~3.08M (trainable) |
| **MAE (sample 0)** | 0.032 | 0.059 | 0.044 | TBD |
| **模式坍塌** | 无 | 无 | 无 | 无 |
| **适用场景** | 稀疏/任意查询点 | 稀疏/任意查询点 | 全图密集预测 | Attention 特征匹配 |

### 3.1 Route D：CNN+ViT Transformer DIC

Route D 将 DIC 形式化为 token-in/token-out 的 Transformer 算子学习问题：

- **CNN Stem**（trainable, 1.2M）：4 级 stride-2 下采样 (256→16×16)，等效 token 分辨率 4×4 px，保留散斑细粒度纹理
- **ViT Encoder**（frozen, 12 layers）：ImageNet 预训练 ViT-base，提供全局 self-attention 上下文
- **Cross-Attention Decoder**（trainable, 1.86M）：RoPE query encoding → CrossAttn(ref_KV) + CrossAttn(tar_KV) → f_diff → MLP fusion
- **双尺度输出**：Coarse 33 bins (log-spaced, ±20px, Cross-Entropy) + Fine 64 bins (uniform, ±0.5px, MSE) → u = u_coarse + Δu
- **复合 Loss**：L_MSE + L_CE(coarse) + λ·L_MSE(fine)，λ=10

设计动机：Attention 的 softmax(QK^T) 在数学上与 DIC 的 correlation-based matching (ZNCC, IC-GN) 同构。

### 3.2 Route A 的位移尺度敏感性

Route A 在**纯小位移数据集**上表现最优（MAE 0.032），但在**混合大小位移的数据集**上出现性能下降。原因分析：

1. **diff 通道的尺度依赖**：Route A 将 `[ref, tar, tar-ref]` 3ch 拼接后送入 CNN，diff 通道的值直接取决于位移大小。大位移样本（diff ≈ 0.3~0.5）主导了 GroupNorm 的统计量，小位移样本（diff ≈ 0.02~0.05）的信号被归一化淹没。

2. **训练梯度冲突**：共享 CNN 权重在同一个 batch 中被大小位移样本的不同梯度方向拉扯，导致两个尺度上的表现均退化。

3. **Route B 的优势**：孪生结构独立编码 ref/tar，diff 在 256 维特征空间计算——编码器本身不受位移尺度影响，天然解耦。

| 数据集 | Route A | Route B | Route C |
|--------|---------|---------|---------|
| 小位移 (< 1px) | **0.032** | 0.059 | 0.044 |
| 混合位移 (0.1~20px) | 退化 | 稳定 | 稳定 |

## 4. 数据准备

### 4.1 准备散斑原图

将真实散斑图案放入 `dataset/original_image/`（支持 png / jpg / bmp / tiff）。

```
dataset/original_image/
  speckle_001.png
  speckle_002.png
  ...
```

### 4.2 编辑数据集配置 `config/dataset.yaml`

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

### 4.3 生成数据集

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

### 4.4 CLI 参数一览

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

## 5. 变形模式说明

| 模式 | 位移范围 | 公式 | 特点 |
|---|---|---|---|
| `tension` | 0.1 ~ 20 px | `u_x = A(x-0.5)`, `u_y = -0.3A(y-0.5)` | x 方向拉伸 + 泊松收缩 |
| `compression` | 0.1 ~ 20 px | 同上取负 | x 方向压缩 |
| `shear` | 0.1 ~ 20 px | `u_x = A(y-0.5)`, `u_y = 0` | 简单剪切 |
| `rotation` | 0.1 ~ 20 px | 绕中心旋转 | 刚体旋转 |
| `composite` | 0.1 ~ 20 px | 拉伸+剪切+正弦非线性 | 复合变形 |
| `multiscale_random` | 0.3 ~ 1.0 px | 多尺度随机控制点 + bicubic 插值 | 复杂亚像素场，边界置零 |

## 6. 训练

### 6.1 简易训练脚本 `_train_simple.py`

支持 Route A / B / C / D，自动保存 best.pt / last.pt。

```powershell
# Route A（默认，batch_size=8）
conda run -n dic python _train_simple.py

# Route B（需减小 batch_size）
conda run -n dic python _train_simple.py --route B --steps 5000 --batch_size 4

# Route C（U-Net，batch_size=4）
conda run -n dic python _train_simple.py --route C --steps 5000 --batch_size 4

# Route D（CNN+ViT Transformer，batch_size=4，复合 loss）
conda run -n dic python _train_simple.py --route D --epochs 100 --batch_size 4

# 自定义参数
conda run -n dic python _train_simple.py --route A --steps 10000 --lr 1e-4 --batch_size 8 --data_dir dataset/dataset/2026-05-27/train
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--route` | A | A / B / C / D |
| `--epochs` | 100 | 训练 epoch 数（Route D 使用 epoch 而非 steps） |
| `--lr` | 1e-4 | 学习率 |
| `--batch_size` | 8 | 批大小（Route B/C/D 建议 4） |
| `--data_dir` | dataset/dataset/2026-06-01/train | 训练数据目录 |

输出：`checkpoints/route_{a,b,c,d}/{best.pt, last.pt}`

### 6.2 完整训练脚本 `train.py`

支持 YAML 配置、checkpoint 断点续训、DDP 多卡。

```powershell
# Route A
conda run -n dic python -m dic_solver_operator.train --config config/training.yaml
conda run -n dic python -m dic_solver_operator.train --resume checkpoints/route_a/last.pt
conda run -n dic python -m dic_solver_operator.train --dataset_dir dataset/dataset/2026-05-27/train

# Route B
conda run -n dic python -m deformation_inverse_operator.train --config config/training.yaml
conda run -n dic python -m deformation_inverse_operator.train --resume checkpoints/route_b/last.pt

# Route C
conda run -n dic python -m dic_unet_method.train --config config/training.yaml
conda run -n dic python -m dic_unet_method.train --dataset_dir dataset/dataset/2026-05-27/train

# 多 GPU（DDP）
torchrun --nproc_per_node=4 -m dic_solver_operator.train --use_ddp --config config/training.yaml
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | - | YAML 配置文件 |
| `--resume` | - | 从 checkpoint 续训 |
| `--dataset_dir` | - | 文件夹数据集路径（覆盖 YAML） |
| `--use_ddp` | False | 启用多卡分布式训练 |

## 7. 预测与可视化

### 7.1 命令行脚本 `_predict.py`

加载训练好的模型，在完整 256×256 网格上预测位移场，生成 3×3 对比图。

```powershell
# 单样本预测
conda run -n dic python _predict.py --route A --ckpt checkpoints/route_a/best.pt --sample 0
conda run -n dic python _predict.py --route B --ckpt checkpoints/route_b/best.pt --sample 0
conda run -n dic python _predict.py --route C --ckpt checkpoints/route_c/best.pt --sample 0
conda run -n dic python _predict.py --route D --ckpt checkpoints/route_d/best.pt --sample 0

# 指定输出路径
conda run -n dic python _predict.py --route C --ckpt checkpoints/route_c/best.pt --sample 0 --save_plot checkpoints/route_c/pred_0.png

# 不传 --save_plot 时自动存到 predictions/route_X_000000.png
conda run -n dic python _predict.py --route A --ckpt checkpoints/route_a/best.pt --sample 5

# 批量预测（PowerShell）
foreach ($s in 0,1,2,3,5,10) { conda run -n dic python _predict.py --route C --ckpt checkpoints/route_c/best.pt --sample $s }
foreach ($s in 0..99) { conda run -n dic python _predict.py --route C --ckpt checkpoints/route_c/best.pt --sample $s }
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--route` | A | A / B / C / D |
| `--ckpt` | - | checkpoint 路径（不存在时自动训练） |
| `--sample` | 0 | 样本编号 |
| `--data_dir` | dataset/dataset/2026-05-27/test | 数据目录 |
| `--save_plot` | auto | 图片保存路径（默认 `predictions/route_X_000000.png`） |

### 7.2 Python API `predict.py`

各路线提供一致的 `Predictor` API，适用于批量推理和集成。

```python
# Route A — 算子式：encode → decode 分离，支持任意查询点
import dic_solver_operator.predict as PA

pred = PA.Predictor("checkpoints/route_a/best.pt")
u_dense = pred.dense(ref_img, tar_img)          # [H, W, 2]
u_sparse = pred.sparse(ref_img, tar_img, pts)    # [N, 2]

# 编码一次、解码多次（高效批量查询）
f_enc = pred.encode(ref_img, tar_img)
u1 = pred.decode(points_a, f_enc)
u2 = pred.decode(points_b, f_enc)

# Route B — 同样支持 encode/decode 分离
import deformation_inverse_operator.predict as PB

pred = PB.Predictor("checkpoints/route_b/best.pt")
u_dense = pred.dense(ref_img, tar_img)
u_sparse = pred.sparse(ref_img, tar_img, pts)

# Route C — 密集预测，一次前向全图输出
import dic_unet_method.predict as PC

pred = PC.Predictor("checkpoints/route_c/best.pt")
u_dense = pred.dense(ref_img, tar_img)          # [H, W, 2]

# Route D — CNN+ViT Transformer，支持 encode/decode 分离
import dic_vit_method.predict as PD

pred = PD.Predictor("checkpoints/route_d/best.pt")
u_dense = pred.dense(ref_img, tar_img)          # [H, W, 2]
u_sparse = pred.sparse(ref_img, tar_img, pts)   # [N, 2]

# 一次性调用（不复用 Predictor）
u = PA.predict_dense(ref_img, tar_img, ckpt="checkpoints/route_a/best.pt")
u = PC.predict_dense(ref_img, tar_img, ckpt="checkpoints/route_c/best.pt")
```

输入：
- `ref_img` / `tar_img` — `np.ndarray [H, W]`，值域 `[0, 1]`（float32）
- `query_points` — `np.ndarray [N, 2]`，归一化坐标 `[0, 1]²`（x, y 顺序，Route C 不支持）

返回：
- `u` — `np.ndarray [H, W, 2]` 或 `[N, 2]`，位移单位为**像素**

## 8. 不规则 ROI 测试 `_test_irregular_roi.py`

将参考图 ROI 外区域置黑，用 GT 位移场 warp 生成变形图，同时测试 Route A / B / C / D 四条路线。

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
| `--ckpt_c` | checkpoints/route_c/best.pt | Route C checkpoint |
| `--ckpt_d` | checkpoints/route_d/best.pt | Route D checkpoint |
| `--save_plot` | predictions/irregular_roi.png | 结果图片路径 |

不规则 ROI 测试结果（sample 0, 圆形 ROI）：

| Route | MAE | MSE | ratio |
|-------|------|------|-------|
| A | 0.0272 | 0.0012 | 0.018 (57x) |
| B | 0.0396 | 0.0026 | 0.039 (26x) |
| C | 0.0566 | 0.0049 | 0.072 (14x) |

### 结果图

**数据集自带 ROI（全图）：**

![irregular_roi_abc](predictions/irregular_roi_abc.png)

**环形 ROI：**

![irregular_roi_ring_abc](predictions/irregular_roi_ring_abc.png)

**缺口板 ROI（dogbone + 中心孔）：**

![irregular_roi_notch_abc](predictions/irregular_roi_notch_abc.png)

## 9. FEM 位移场不规则 ROI 测试 `test/`

基于 **2D 线弹性 Navier-Cauchy 方程有限差分求解** 生成符合真实实验工况的位移场，替代随机位移场进行测试。

### 9.1 测试用例

| Case | 工况 | BC | 位移场 |
|------|------|----|--------|
| circle | 圆盘压缩 | 顶点加载 (u_y=-d, u_x=0) / 底点固支 (u_x=u_y=0) | u_y ∈ [-d, 0], u_x ≈ ±0.5d |
| ring | 圆环压缩 | 同上 | u_y ∈ [-d, 0], u_x ≈ ±0.5d |
| notch | 缺口板拉伸 | 顶边加载 (u_y=+d, u_x=0) / 底边固支 (u_x=u_y=0) | u_y ∈ [0, +d], u_x ≈ ±0.5d |
| full | 全图压缩（基准） | 同上（圆盘） | u_y ∈ [-d, 0], u_x ≈ ±0.5d |

### 9.2 快速使用

```powershell
# 全部测试
python -m test.run --all

# 单工况
python -m test.run --case circle
python -m test.run --case ring
python -m test.run --case notch
python -m test.run --case full

# 自定义参数
python -m test.run --case circle --disp_range -1 1 --seed 42

# 仅生成 / 仅预测
python -m test.run --case circle --generate-only
python -m test.run --case circle --predict-only
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--all` | - | 运行全部 4 个工况 |
| `--case` | - | circle / ring / notch / full |
| `--disp_range` | -4 4 | 位移加载范围 (px) |
| `--image_size` | 256 | 图像尺寸 |
| `--seed` | 0 | 随机种子 |
| `--generate-only` | False | 仅生成图像 |
| `--predict-only` | False | 仅预测 |
| `--speckle_source` | test/speckle.png | 散斑图路径 |

### 9.3 位移加载 1px 测试结果

| Case | ROI | u_y | Route A | Route B | Route C | Route D |
|------|-----|-----|---------|---------|---------|---------|
| full | 99% | [-1, 0] | 0.027 | **0.021** | 0.022 | TBD |
| circle | 38% | [-1, 0] | 0.031 | **0.022** | 0.033 | TBD |
| ring | 44% | [-1, 0] | 0.030 | **0.021** | 0.033 | TBD |
| notch | 26% | [0, 1] | 0.042 | **0.028** | 0.037 | TBD |

**关键发现**：
- **Route B 全部工况最优**——孪生编码器 + 特征空间差分对 FEM 位移模式最鲁棒
- 全图基准 MAE 最低 (0.021)，不规则 ROI 和几何不连续（孔、缺口）是主要误差源
- Route A 在 FEM 光滑场上不如随机训练场——其 diff 通道被"平滑"位移模式削弱

### 9.4 文件结构

```
test/
  speckle.png                  # 散斑纹理源
  roi_shapes.py                # ROI 形状生成
  fem_displacement.py          # 2D Navier-Cauchy 有限差分求解器
  generate.py                  # 图像生成管线
  predict.py                   # 多路线预测 + 可视化
  run.py                       # CLI 入口
  output/
    circle/                    # ref.png, tar.png, u_field.npy, roi_mask.png, *_results.png
    ring/
    notch/
    full/
```

## 10. 损失函数

### 简易训练 `_train_simple.py`

**Route A / B / C** 统一使用 **MSE Loss**：

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\text{ROI}|} \sum_{i \in \text{ROI}} \| \hat{u}_i - u_i \|^2$$

- Route A / B：在模型输出的查询点 `[B, N_q, 2]` 上直接计算
- Route C：从 UNet 密集输出 `[B, 2, H, W]` 经 `grid_sample` 采样到查询点后计算

**Route D** 使用 **复合 Loss**（详见 `reference_paper/route_d_design.md`）：

$$\mathcal{L} = \mathcal{L}_{\text{MSE}}(\hat{u}, u) + \mathcal{L}_{\text{CE}}(\text{coarse}_x, \text{bin}(u_x)) + \mathcal{L}_{\text{CE}}(\text{coarse}_y, \text{bin}(u_y)) + \lambda \cdot \mathcal{L}_{\text{MSE}}(\Delta u, u - u_{\text{coarse}})$$

- Coarse head: 33 个 log-spaced bins (±20px)，Cross-Entropy 分类
- Fine head: 64 个 uniform bins (±0.5px)，MSE 回归亚像素残差
- λ = 10（精细回归权重）

### 完整训练 `train.py`

各路线使用 `CompositeLoss`（`common/losses.py`），支持可配置的数据损失和正则化：

| 数据损失 | 公式 | 说明 |
|---------|------|------|
| `huber` | 分段 L1/L2，δ=1.0 | 对大误差更鲁棒（当前默认） |
| `relative_l2` | $\| \hat{u} - u \|_2 / \| u \|_2$ | 相对误差 |

| 正则化 | 状态 | 说明 |
|--------|------|------|
| `smoothness` | 未实现 (λ=0) | 惩罚位移场梯度剧烈变化 |
| `compatibility` | 未实现 (λ=0) | 惩罚应变不兼容 |

## 11. 架构说明

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

编码器在第一个卷积层直接比较 ref/tar/diff，局部特征采样解码。适合任意查询点、稀疏推理。

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

孪生 CNN 分别编码参考图和目标图，解码器通过比较两个特征图的局部差异推断位移。

### Route C：U-Net Dense Prediction

```
Ref + Tar (2ch)
       │
  ┌─────────────────────────┐
  │  Encoder                │
  │  conv1(64) → pool       │ 256
  │  conv2(128) → pool      │ 128
  │  conv3(256) → pool      │ 64
  │  conv4(512) → pool      │ 32
  │  bottleneck(1024)       │ 16
  │                         │
  │  Decoder (skip connect) │
  │  up4+conv(512) → up3+conv(256)  │
  │  → up2+conv(128) → up1+conv(64) │
  │  → head Conv2d → u [2, H, W]   │
  └─────────────────────────┘
```

标准 5 层 U-Net，输入 ref+tar 拼接，一次前向输出全分辨率位移场。使用 GroupNorm + GELU，无 BatchNorm。

### Route D：CNN + ViT Transformer DIC

```
ref [B,1,256,256]
       │
  CNN Stem (trainable, 1.2M)          ← 4级 stride-2 下采样
   ├─ Conv7×7 + GN + GELU             ← 7×7 捕捉 2~5px 颗粒
   ├─ ResBlock(64→64, /2)  → 128×128
   ├─ ResBlock(64→96, /2)  → 64×64
   ├─ ResBlock(96→128, /2) → 32×32
   ├─ ResBlock(128→192, /2) → 16×16
   └─ Conv1×1(192→768)      → 256 tokens × 768-dim
       │
  ViT Encoder (frozen, 12 layers)     ← 全局 self-attention
       │
  Proj(768→256) + RoPE
       │
  Cross-Attention Decoder (trainable, 1.86M)
   ├─ RoPE query encoding
   ├─ CrossAttn(Q→ref_KV) → q_ref
   ├─ CrossAttn(Q→tar_KV) → q_tar
   ├─ f_diff = q_tar - q_ref
   ├─ MLP fusion
   └─ Dual-scale heads (33 + 64 bins) → u_pred [B, N_q, 2]
```

CNN stem 与 Route A/B 共用 ResBlock + GN + GELU 模式，4 级下采样后等效 token 分辨率 4×4 px（优于 ViT 原生 16×16 px）。仅 3.08M 可训练参数。

### 设计要点

- **Route D Attention 匹配**：Cross-attention 的 softmax(QK^T) 在数学上与 DIC correlation-based matching (ZNCC, IC-GN) 同构
- **无交叉注意力**（Route A/B）：使用双线性特征采样 + MLP 解码，避免 Galerkin 交叉注意力的 K^T@V 瓶颈导致的模式坍塌
- **GFF 位置编码**（Route A/B）：高斯傅里叶特征编码查询坐标，使 MLP 能表示高频位移变化
- **密集预测**（Route C）：U-Net 直接输出全图，无需查询点采样，训练时从密集输出采样到查询点计算损失
- **统一损失框架**：Route A/B/C 共用 MSE / Huber 损失；Route D 使用 MSE + CE(coarse) + λ·MSE(fine) 复合损失，仅在 ROI 内有效点计算

## 12. 项目结构

```
├── common/                         # 共享组件
│   ├── cross_attention.py          # Galerkin 线性交叉注意力
│   ├── self_attention.py           # 线性自注意力
│   ├── gaussian_fourier_features.py # GFF 坐标编码
│   ├── feedforward.py              # FFN
│   ├── layer_norm.py               # LayerNorm / PostNorm
│   ├── losses.py                   # CompositeLoss (Huber / Relative L2)
│   ├── metrics.py                  # Relative L2, Huber 计算
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
├── dic_solver_operator/            # Route A：算子学习
│   ├── config.py                   # SolverOperatorConfig
│   ├── encoder.py                  # DualChannelCNN
│   ├── decoder.py                  # SimpleLocalDecoder
│   ├── model.py                    # SolverOperatorModel
│   ├── predict.py                  # 预测 API
│   └── train.py                    # 完整训练脚本
├── deformation_inverse_operator/   # Route B：逆算子学习
│   ├── config.py                   # InverseOperatorConfig
│   ├── encoder.py                  # SiameseCNN
│   ├── decoder.py                  # InverseDecoder
│   ├── model.py                    # InverseOperatorModel
│   ├── predict.py                  # 预测 API
│   └── train.py                    # 完整训练脚本
├── dic_unet_method/                # Route C：U-Net 密集预测
│   ├── config.py                   # UnetDICConfig
│   ├── model.py                    # UnetDICModel
│   ├── predict.py                  # 预测 API
│   └── train.py                    # 完整训练脚本
├── dic_vit_method/                 # Route D：CNN+ViT Transformer
│   ├── config.py                   # VitDICConfig (coarse/fine bins, CNN stem)
│   ├── encoder.py                  # CNNStem + ViTEncoder (混合架构)
│   ├── decoder.py                  # TransformerDecoder + DisplacementHead
│   ├── model.py                    # VitDICModel (encode/decode/forward)
│   └── predict.py                  # 预测 API
├── config/
│   ├── dataset.yaml                # 数据集生成配置
│   └── training.yaml               # 训练配置
├── checkpoints/
│   ├── route_a/                    # Route A 模型权重
│   ├── route_b/                    # Route B 模型权重
│   ├── route_c/                    # Route C 模型权重
│   └── route_d/                    # Route D 模型权重
├── predictions/                    # 预测结果图片
├── test/                            # FEM 位移场不规则 ROI 测试
│   ├── speckle.png                  # 散斑纹理源
│   ├── roi_shapes.py                # ROI 形状生成 (circle/ring/notch/full)
│   ├── fem_displacement.py          # 2D Navier-Cauchy 有限差分求解器
│   ├── generate.py                  # 图像生成管线
│   ├── predict.py                   # 多路线预测 + 可视化
│   ├── run.py                       # CLI 入口
│   └── output/                      # 生成结果
├── _train_simple.py                # 简易训练脚本 (A/B/C/D)
├── _predict.py                     # 预测 + 可视化脚本 (A/B/C/D)
├── _test_irregular_roi.py          # 不规则 ROI 测试 (A/B/C/D)
├── _test_multires.py               # 多分辨率测试 (A/B/C/D)
├── reference_paper/                # 设计文档
│   └── route_d_design.md           # Route D 架构设计记录
└── experiments/                    # 实验配置和启动脚本
```

## 13. 技术要点

- **Galerkin 线性注意力**（历史）：无 softmax，O(N·d²) 复杂度，K 列 InstanceNorm 保证基函数单位范数。当前 Route A/B 已移除交叉注意力，改用局部特征采样解码器
- **模式坍塌修复**：移除 V InstanceNorm + 最终替换为 bilinear 局部特征采样 + MLP 解码
- **GFF 位置编码**：高斯傅里叶特征将坐标映射到高频空间，使浅层 MLP 能够表示复杂的非线性位移场
- **正映射 splatting**（数据生成）：参考图像素按位移推至目标位置 + 双线性权重分散
- **ROI 自动计算**：逐像素判定 `x + u(x)` 是否在图像边界内；预测时仅在有效区域评估
