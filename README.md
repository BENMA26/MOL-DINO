# MOL-DINO: Molecule DINO for 3D Generation

将预训练的Molecule DINO encoder应用于3D分子生成任务，在QM9数据集上使用UniGEM benchmark评估。

[![GitHub](https://img.shields.io/badge/GitHub-MOL--DINO-blue)](https://github.com/BENMA26/MOL-DINO)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 项目概述

本项目将3D-GSRD的Molecule DINO预训练encoder接入Tabasco生成框架，验证预训练3D几何表征对分子生成质量的提升效果。

### 核心创新

- **预训练Encoder**: 使用DINO自蒸馏预训练的3D几何encoder
- **两阶段训练**: Stage 1冻结encoder训练decoder，Stage 2端到端微调
- **Flow Matching**: 在latent space使用flow matching生成分子
- **UniGEM Benchmark**: 使用atom/molecule stability等严格指标评估

### 数据集与评估

- **数据集**: QM9 (~130k小分子，5种原子类型: H, C, N, O, F)
- **评估指标**:
  - Atom Stability: 每个原子化学价是否合理
  - Molecule Stability: 整个分子化学价是否全部合理
  - Validity: RDKit能否成功解析
  - Uniqueness: 去重比例
  - Novelty: 训练集未见过的比例

## 🚀 快速开始

### 环境要求

```bash
conda activate mol_dino
```

### 一键启动实验

```bash
cd code/tabasco/src
bash submit_all_jobs.sh
```

这会自动提交3个作业（Stage 1 → Stage 2 → Stage 3），并设置依赖关系。

### 监控进度

```bash
# 查看作业状态
qstat -u $USER

# 实时查看日志
tail -f dino_ae_stage1.log
tail -f dino_ae_stage2.log
tail -f dino_flow.log
```

## 📊 实验流程

### Stage 1: 冻结Encoder训练Decoder (100 epochs)

让decoder学会从DINO encoder的表征重建分子。

```bash
qsub train_dino_ae_stage1_pbs.sh
```

**超参数**: batch_size=64, lr=1e-4, freeze_encoder=True

### Stage 2: 解冻Encoder端到端微调 (50 epochs)

微调encoder使其更适配生成任务。

```bash
qsub train_dino_ae_stage2_pbs.sh
```

**超参数**: batch_size=32, lr=5e-5, freeze_encoder=False

### Stage 3: Flow Matching训练 (200 epochs)

在latent space学习分子分布。

```bash
qsub train_dino_flow_pbs.sh
```

**超参数**: batch_size=128, lr=1e-4

### Stage 4: 生成评估

采样1000个分子，计算UniGEM benchmark指标。

```bash
python eval_dino_gen.py \
    --ae_ckpt outputs/dino_ae_qm9_stage2/best.ckpt \
    --flow_ckpt outputs/dino_flow_qm9/best.ckpt \
    --ref_data_dir /path/to/qm9_train.pt \
    --n_samples 1000 \
    --num_steps 100 \
    --batch_size 64
```

## 🏗️ 架构设计

```
Input Molecule (coords, atomics, padding_mask)
    ↓
[DINO Encoder] ← 预训练冻结，来自3D-GSRD
    ↓ PyG sparse graph
Node Features (B, N, 256)
    ↓
[VAE Bottleneck] ← per-node reparameterization (kl_dim=6)
    ↓
Latent z (B, N, 6)
    ↓
[Transformer Decoder] ← 可训练 (8 layers, 8 heads)
    ↓
Reconstructed Molecule
    ↓
[Flow Matching] ← 在latent space生成
    ↓
Generated Molecule
```

### 关键组件

1. **DinoEncoderModule** (`code/tabasco/src/tabasco/models/dino_encoder.py`)
   - Dense ↔ PyG稀疏图转换
   - 加载预训练的3D-GSRD encoder
   - Per-node VAE bottleneck
   - Transformer decoder

2. **两阶段训练策略**
   - Stage 1: 冻结encoder，让decoder适配预训练表征
   - Stage 2: 解冻encoder，端到端优化

3. **Flow Matching生成**
   - OT interpolant (coords)
   - VP interpolant (atomics)
   - Euler采样

## 📁 项目结构

```
MOL-DINO/
├── code/
│   ├── 3D-GSRD/                    # Molecule DINO预训练代码
│   ├── tabasco/src/                # 生成实验代码
│   │   ├── tabasco/models/
│   │   │   └── dino_encoder.py     # 核心适配模块
│   │   ├── train_dino_ae.py        # AE训练脚本
│   │   ├── train_dino_flow.py      # Flow训练脚本
│   │   ├── eval_dino_gen.py        # 生成评估脚本
│   │   ├── train_dino_ae_stage1_pbs.sh
│   │   ├── train_dino_ae_stage2_pbs.sh
│   │   ├── train_dino_flow_pbs.sh
│   │   ├── submit_all_jobs.sh      # 一键提交脚本
│   │   └── README_DINO_GENERATION.md
│   ├── UniGEM/                     # UniGEM评估代码
│   ├── EXPERIMENT_GUIDE.md         # 详细实验指南
│   ├── QM9_EXPERIMENT_SUMMARY.md   # QM9实验总结
│   └── QUICKSTART.md               # 快速启动指南
├── materials/                      # 论文材料
└── notes/                          # 项目笔记
```

## 📈 预期结果

### UniGEM Baseline (QM9)

根据UniGEM论文Table 1：
- Atom Stability: ~0.95-0.98
- Molecule Stability: ~0.85-0.90
- Validity: ~0.90-0.95

### DINO版本目标

由于使用了预训练的3D几何encoder：
- **Atom Stability**: ≥0.96 (理解化学价)
- **Molecule Stability**: ≥0.88 (整体结构合理)
- **Validity**: ≥0.92 (符合化学规则)
- **Uniqueness**: ≥0.95 (多样性)
- **Novelty**: ≥0.90 (泛化能力)

## 📚 文档

- [完整README](code/tabasco/src/README_DINO_GENERATION.md) - 详细技术文档
- [实验指南](code/EXPERIMENT_GUIDE.md) - 完整实验流程
- [QM9实验总结](code/QM9_EXPERIMENT_SUMMARY.md) - QM9实验详情
- [快速启动](code/QUICKSTART.md) - 快速上手指南

## 🔧 技术细节

### Dense ↔ PyG转换

DINO encoder需要PyG稀疏图，Tabasco使用dense tensor。`dense_to_pyg_batch()`实现自动转换：
- 输入: (B, N, 3) coords + (B, N, A) atomics
- 输出: PyG Batch with full-graph edges

### VAE Bottleneck

Per-node VAE reparameterization:
- Input: (B, N, hidden_dim=256)
- Output: (B, N, kl_dim=6)
- KL weight: 1e-6

### 超参数配置

**Autoencoder**:
- gsrd_hidden_dim: 256
- hidden_dim: 256
- kl_dim: 6
- num_heads: 8
- num_layers: 8

**Flow Matching**:
- hidden_dim: 256
- num_heads: 8
- num_layers: 6
- batch_size: 128

## 🐛 故障排查

### 常见问题

1. **维度不匹配**: 检查`node_dim=25` (22 atom types + 3 coords)
2. **Checkpoint加载失败**: 确认路径和key前缀`model._orig_mod.encoder.*`
3. **OOM错误**: 降低batch_size
4. **训练不收敛**: 检查kl_weight和学习率

详见[实验指南](code/EXPERIMENT_GUIDE.md)的故障排查章节。

## 📖 参考文献

- **3D-GSRD**: Selective Re-mask Decoding for 3D molecular representation
- **DINO**: Self-distillation with no labels
- **UniGEM**: Unified generative model for molecules
- **Tabasco**: Flow matching for 3D molecule generation

## 📝 License

MIT License

## 🙏 致谢

本项目基于以下开源项目：
- [3D-GSRD](https://github.com/example/3D-GSRD) - Molecule DINO预训练
- [Tabasco](https://github.com/example/tabasco) - 3D分子生成框架
- [UniGEM](https://github.com/example/UniGEM) - 生成评估benchmark

## 📧 联系方式

如有问题，请提交Issue或查看详细文档。

---

**预计实验时间**: 3-5天（取决于集群负载）

**下一步**: 提交作业后，定期检查日志文件，确保训练正常进行。
