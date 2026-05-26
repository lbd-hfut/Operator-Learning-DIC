# Operator-Learning-DIC

基于 Neural Operator（神经算子）的数字图像相关（DIC）位移场测量框架。

核心思路：将 DIC 的灰度不变假设 `I_ref(x) = I_tar(x + u(x))` 形式化为算子学习问题，一次训练后可在任意分辨率、任意查询点上推理位移场。

## 1. 环境配置

```bash
conda create -n dic python=3.10
conda activate dic
pip install torch numpy scipy h5py matplotlib tqdm einops PyYAML pillow
```

## 2. 数据准备

### 2.1 准备散斑原图

将真实散斑图案放入 `dataset/original_image/`（支持 png / jpg / bmp / tiff）。

```
dataset/original_image/
  speckle_001.png
  speckle_002.png
  ...
```

### 2.2 编辑数据集配置

`config/dataset.yaml`：

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

### 2.3 生成数据集

```bash
# 使用 YAML 配置生成
python -m dataset.generate_dataset --config config/dataset.yaml

# 覆盖分集数量
python -m dataset.generate_dataset --config config/dataset.yaml --train 100 --test 50
```

输出目录自动以时间戳命名：

```
dataset/dataset/2026-05-26/
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

### 2.4 CLI 参数一览

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

## 3. 变形模式说明

| 模式 | 位移范围 | 公式 | 特点 |
|---|---|---|---|
| `tension` | 0.1 ~ 20 px | `u_x = A(x-0.5)`, `u_y = -0.3A(y-0.5)` | x 方向拉伸 + 泊松收缩 |
| `compression` | 0.1 ~ 20 px | 同上取负 | x 方向压缩 |
| `shear` | 0.1 ~ 20 px | `u_x = A(y-0.5)`, `u_y = 0` | 简单剪切 |
| `rotation` | 0.1 ~ 20 px | 绕中心旋转 | 刚体旋转 |
| `composite` | 0.1 ~ 20 px | 拉伸+剪切+正弦非线性 | 复合变形 |
| `multiscale_random` | 0.3 ~ 1.0 px | 多尺度随机控制点 + bicubic 插值 | 复杂亚像素场，边界置零 |

## 4. 训练

### Route A：DIC Solver Operator

```bash
python -m dic_solver_operator.train \
    --dataset_dir dataset/dataset/2026-05-26/train
```

架构：双通道 CNN 编码 [I_ref, I_tar] → 交叉注意力解码查询坐标 → u

### Route B：Deformation Inverse Operator

```bash
python -m deformation_inverse_operator.train \
    --dataset_dir dataset/dataset/2026-05-26/train
```

架构：孪生 CNN 分别编码 I_ref / I_tar → 差分交叉注意力压缩为 latent tokens → 交叉注意力解码查询坐标 → u

### 多 GPU（DDP）

```bash
torchrun --nproc_per_node=4 -m dic_solver_operator.train --use_ddp --dataset_dir ...
```

## 5. 项目结构

```
├── common/                         # 共享组件（注意力、位置编码、损失、存档）
├── dataset/                        # 数据管线
│   ├── image_pool.py               # 真实散斑图加载
│   ├── deformation_generator.py    # 位移场生成（6 种模式）
│   ├── warp.py                     # 正映射 splatting
│   ├── roi.py                      # ROI 计算（逐像素有效掩码）
│   ├── generate_dataset.py         # 离线数据集生成 CLI
│   ├── folder_dataset.py           # 训练时读取文件夹数据集
│   └── ...
├── dic_solver_operator/            # Route A 模型 + 训练
├── deformation_inverse_operator/   # Route B 模型 + 训练
├── config/
│   └── dataset.yaml                # 数据集生成配置
└── experiments/                    # 实验配置和启动脚本
```

## 6. 技术要点

- **Galerkin 线性注意力**：无 softmax，O(N·d²) 复杂度，K/V 列 InstanceNorm 保证基函数单位范数
- **正映射 splatting**：参考图像素按位移推至目标位置 + 双线性权重分散，位移场在正确坐标帧读取
- **ROI 自动计算**：逐像素判定 `x + u(x)` 是否在图像边界内，训练时仅在有效区域采样查询点
