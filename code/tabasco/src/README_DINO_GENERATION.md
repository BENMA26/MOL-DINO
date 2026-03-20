# DINO Encoder for 3D Molecule Generation

将预训练的Molecule DINO encoder接入Tabasco生成框架，在QM9数据集上进行3D分子生成实验。

## 📋 目录

- [项目概述](#项目概述)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [实验流程](#实验流程)
- [评估指标](#评估指标)
- [文件结构](#文件结构)
- [技术细节](#技术细节)

## 项目概述

### 目标

验证预训练的3D几何encoder（Molecule DINO）对分子生成质量的提升效果。

### 数据集

**QM9**: 包含约130k个小分子，5种原子类型（H, C, N, O, F）

### 评估基准

使用**UniGEM benchmark**评估指标：
- **Atom Stability**: 每个原子的化学价是否合理
- **Molecule Stability**: 整个分子的化学价是否全部合理
- **Validity**: RDKit能否成功解析分子
- **Uniqueness**: 生成分子去重后的比例
- **Novelty**: 训练集中未见过的分子比例

## 架构设计

### 整体流程

```
Input Molecule
    ↓
[DINO Encoder] ← 预训练冻结，来自3D-GSRD
    ↓
Node Features (B, N, 256)
    ↓
[VAE Bottleneck] ← per-node reparameterization
    ↓
Latent z (B, N, 6)
    ↓
[Transformer Decoder] ← 可训练
    ↓
Reconstructed Molecule
```

### 关键组件

1. **DinoEncoderModule** (`tabasco/models/dino_encoder.py`)
   - Dense ↔ PyG稀疏图转换
   - 加载预训练的3D-GSRD encoder
   - Per-node VAE bottleneck (kl_dim=6)
   - Transformer decoder (8 layers, 8 heads)

2. **两阶段训练**
   - **Stage 1**: 冻结encoder，训练decoder (100 epochs)
   - **Stage 2**: 解冻encoder，端到端微调 (50 epochs)

3. **Flow Matching**
   - 在latent space训练flow matching模型 (200 epochs)
   - 使用OT interpolant (coords) + VP interpolant (atomics)

## 快速开始

### 前置要求

```bash
# 环境
conda activate mol_dino

# 数据路径（需要提前准备）
/scratch/yuxuan.ren/maben/data/qm9/
├── qm9_train.pt
└── qm9_val.pt

# Checkpoint路径
/scratch/yuxuan.ren/maben/checkpoints/pretrain.ckpt
```

### 一键提交所有作业

```bash
cd /scratch/yuxuan.ren/maben/code/tabasco/src
bash submit_all_jobs.sh
```

这会自动提交3个作业，并设置依赖关系：
- Stage 1 → Stage 2 → Stage 3

### 监控作业

```bash
# 查看作业状态
qstat -u $USER

# 查看日志
tail -f dino_ae_stage1.log
tail -f dino_ae_stage2.log
tail -f dino_flow.log
```

## 实验流程

### Stage 1: 冻结Encoder训练Decoder

**目的**: 让decoder学会从DINO encoder的表征重建分子

```bash
qsub train_dino_ae_stage1_pbs.sh
```

**超参数**:
- Batch size: 64
- Learning rate: 1e-4
- Epochs: 100
- Freeze encoder: ✓

**输出**:
- Checkpoint: `outputs/dino_ae_qm9_stage1/best.ckpt`
- 日志: `dino_ae_stage1.log`

**监控指标**:
- `train/coord_loss`: 坐标重建MSE
- `train/atom_loss`: 原子类型交叉熵
- `train/kl_loss`: KL散度损失
- `val/loss`: 验证集总损失

---

### Stage 2: 解冻Encoder端到端微调

**目的**: 微调encoder使其更适配生成任务

```bash
qsub train_dino_ae_stage2_pbs.sh  # 自动等待Stage 1完成
```

**超参数**:
- Batch size: 32
- Learning rate: 5e-5 (降低)
- Epochs: 50
- Freeze encoder: ✗
- Resume from: Stage 1 checkpoint

**输出**:
- Checkpoint: `outputs/dino_ae_qm9_stage2/best.ckpt`
- 日志: `dino_ae_stage2.log`

---

### Stage 3: Flow Matching训练

**目的**: 在latent space学习分子分布

```bash
qsub train_dino_flow_pbs.sh  # 自动等待Stage 2完成
```

**超参数**:
- Batch size: 128
- Learning rate: 1e-4
- Epochs: 200
- Frozen AE: ✓

**输出**:
- Checkpoint: `outputs/dino_flow_qm9/best.ckpt`
- 日志: `dino_flow.log`

**监控指标**:
- `train/loss`: flow matching损失
- `val/loss`: 验证集损失

---

### Stage 4: 生成评估

**目的**: 采样1000个分子，计算UniGEM benchmark指标

```bash
python eval_dino_gen.py \
    --ae_ckpt outputs/dino_ae_qm9_stage2/best.ckpt \
    --flow_ckpt outputs/dino_flow_qm9/best.ckpt \
    --ref_data_dir /scratch/yuxuan.ren/maben/data/qm9/qm9_train.pt \
    --lmdb_dir /scratch/yuxuan.ren/maben/lmdb_cache/qm9 \
    --n_samples 1000 \
    --num_steps 100 \
    --batch_size 64
```

**输出示例**:
```
==================================================
QM9 Generation Metrics (UniGEM benchmark)
==================================================
Total generated    : 1000
Atom stability     : 0.9650
Molecule stability : 0.8820
Validity           : 0.9230
Uniqueness         : 0.9580
Novelty            : 0.9120
==================================================
```

## 评估指标

### UniGEM Baseline (QM9)

根据UniGEM论文Table 1：
- Atom Stability: ~0.95-0.98
- Molecule Stability: ~0.85-0.90
- Validity: ~0.90-0.95

### DINO版本预期

由于使用了预训练的3D几何encoder：
- **Atom Stability**: ≥0.96 (理解化学价)
- **Molecule Stability**: ≥0.88 (整体结构合理)
- **Validity**: ≥0.92 (符合化学规则)
- **Uniqueness**: ≥0.95 (多样性)
- **Novelty**: ≥0.90 (泛化能力)

### 指标说明

**Atom Stability**
- 检查每个原子的化学价是否符合规则
- 例如: C应该有4个键，N应该有3个键，O应该有2个键
- 使用UniGEM的`check_stability()`函数，基于原子间距离判断bond order

**Molecule Stability**
- 整个分子的所有原子化学价都合理才算stable
- 更严格的指标，反映整体分子质量

**Validity**
- RDKit能否成功解析并sanitize分子
- 检查化学结构的基本合理性

**Uniqueness**
- 生成的分子去重后的比例
- 反映生成的多样性

**Novelty**
- 生成的分子中训练集未见过的比例
- 反映模型的泛化能力

## 文件结构

```
code/tabasco/src/
├── tabasco/models/
│   └── dino_encoder.py              # 核心适配模块
├── train_dino_ae.py                 # AE训练脚本
├── train_dino_flow.py               # Flow训练脚本
├── eval_dino_gen.py                 # 生成评估脚本
├── train_dino_ae_stage1_pbs.sh      # Stage 1 PBS作业
├── train_dino_ae_stage2_pbs.sh      # Stage 2 PBS作业
├── train_dino_flow_pbs.sh           # Stage 3 PBS作业
├── submit_all_jobs.sh               # 一键提交脚本
└── README_DINO_GENERATION.md        # 本文档
```

## 技术细节

### Dense ↔ PyG转换

DINO encoder需要PyG稀疏图输入，Tabasco使用dense tensor。`dense_to_pyg_batch()`函数实现转换：

```python
def dense_to_pyg_batch(coords, atomics, padding_mask):
    """
    coords: (B, N, 3) dense tensor
    atomics: (B, N, A) one-hot
    padding_mask: (B, N) bool

    Returns: PyG Batch with full-graph edges
    """
```

### 全连接图

Encoder输入是全连接图（所有原子对都有边），edge_attr使用全零占位符，让模型纯靠距离特征学习。

### VAE Bottleneck

Per-node VAE reparameterization:
- Input: (B, N, hidden_dim=256)
- Output: (B, N, kl_dim=6)
- KL weight: 1e-6

### 位置编码

Decoder使用sinusoidal positional encoding（参考MOL-AE设计）：
- 只加在decoder，不加在encoder
- Encoder学到的是与顺序无关的全局表征

### 超参数配置

**Autoencoder**:
- `gsrd_hidden_dim`: 256 (DINO encoder维度)
- `hidden_dim`: 256 (decoder维度)
- `kl_dim`: 6 (latent维度)
- `num_heads`: 8
- `num_layers`: 8 (decoder层数)

**Flow Matching**:
- `hidden_dim`: 256
- `num_heads`: 8
- `num_layers`: 6
- `time_distribution`: uniform

## 故障排查

### 常见问题

**1. 维度不匹配**
```
检查 make_gsrd_args() 中的配置:
- node_dim = 22 + 3 = 25 (atom types + coords)
- edge_dim = 1 (distance only)
需与预训练时一致
```

**2. Checkpoint加载失败**
```
确认路径:
- DINO checkpoint: /scratch/yuxuan.ren/maben/checkpoints/pretrain.ckpt
- Key前缀: model._orig_mod.encoder.*
```

**3. OOM (Out of Memory)**
```
降低batch_size:
- Stage 1: 64 → 32
- Stage 2: 32 → 16
- Flow: 128 → 64
```

**4. 训练不收敛**
```
检查:
- kl_weight是否过大 (默认1e-6)
- 学习率是否合适
- coord_loss和atom_loss是否都在下降
```

**5. UniGEM依赖缺失**
```
评估脚本需要访问UniGEM代码:
- 路径: ../../UniGEM
- 需要: qm9/analyze.py, qm9/bond_analyze.py
```

### 调试技巧

**查看训练日志**:
```bash
# 实时查看
tail -f dino_ae_stage1.log

# 搜索错误
grep -i error dino_ae_stage1.log

# 查看loss曲线
grep "train/loss" dino_ae_stage1.log
```

**检查checkpoint**:
```python
import torch
ckpt = torch.load("outputs/dino_ae_qm9_stage1/best.ckpt")
print(ckpt.keys())
print(ckpt['hyper_parameters'])
```

**测试单个batch**:
```python
from train_dino_ae import DinoAELitModule
model = DinoAELitModule.load_from_checkpoint("best.ckpt")
# 测试forward pass
```

## 参考文献

- **3D-GSRD**: Selective Re-mask Decoding for 3D molecular representation
- **DINO**: Self-distillation with no labels
- **UniGEM**: Unified generative model for molecules
- **Tabasco**: Flow matching for 3D molecule generation

## 联系方式

如有问题，请查看:
- `EXPERIMENT_GUIDE.md`: 完整实验指南
- `QM9_EXPERIMENT_SUMMARY.md`: QM9实验详细总结
- `notes/session_summary.md`: 项目历史记录
