# DINO Encoder + Flow Matching 生成实验指南

## 实验目标

将Molecule DINO预训练encoder接入Tabasco生成框架，在QM9数据集上验证预训练表征对3D分子生成质量的提升，使用UniGEM benchmark评估指标。

## 实验流程

### Stage 1: 冻结Encoder训练Decoder (100 epochs)

**目的**: 让decoder学会从DINO encoder的表征重建分子

```bash
cd /scratch/yuxuan.ren/maben/code/tabasco/src
qsub train_dino_ae_stage1_pbs.sh
```

**预期输出**:
- Checkpoint: `/scratch/yuxuan.ren/maben/outputs/dino_ae_qm9_stage1/best.ckpt`
- 日志: `dino_ae_stage1.log`
- 训练指标: `outputs/dino_ae_qm9_stage1/logs/`

**监控指标**:
- `train/coord_loss`: 坐标重建MSE
- `train/atom_loss`: 原子类型交叉熵
- `val/loss`: 验证集总损失

### Stage 2: 解冻Encoder端到端微调 (50 epochs)

**目的**: 微调encoder使其更适配生成任务

```bash
# 等待Stage 1完成后
qsub train_dino_ae_stage2_pbs.sh
```

**预期输出**:
- Checkpoint: `/scratch/yuxuan.ren/maben/outputs/dino_ae_qm9_stage2/best.ckpt`
- 日志: `dino_ae_stage2.log`

**注意**: 从Stage 1 checkpoint恢复，学习率降低到5e-5

### Stage 3: 训练Flow Matching模型 (200 epochs)

**目的**: 在latent space学习分子分布

```bash
# 等待Stage 2完成后
qsub train_dino_flow_pbs.sh
```

**预期输出**:
- Checkpoint: `/scratch/yuxuan.ren/maben/outputs/dino_flow_qm9/best.ckpt`
- 日志: `dino_flow.log`

**监控指标**:
- `train/loss`: flow matching损失
- `val/loss`: 验证集损失

### Stage 4: 生成评估 (QM9 + UniGEM metrics)

**目的**: 采样1000个分子，计算UniGEM benchmark指标

```bash
python eval_dino_gen.py \
    --ae_ckpt /scratch/yuxuan.ren/maben/outputs/dino_ae_qm9_stage2/best.ckpt \
    --flow_ckpt /scratch/yuxuan.ren/maben/outputs/dino_flow_qm9/best.ckpt \
    --ref_data_dir /scratch/yuxuan.ren/maben/data/qm9/qm9_train.pt \
    --lmdb_dir /scratch/yuxuan.ren/maben/lmdb_cache/qm9 \
    --n_samples 1000 \
    --num_steps 100 \
    --batch_size 64
```

**预期输出** (UniGEM benchmark):
```
==================================================
QM9 Generation Metrics (UniGEM benchmark)
==================================================
Total generated    : 1000
Atom stability     : 0.XXXX  (每个原子的化学价是否合理)
Molecule stability : 0.XXXX  (整个分子的化学价是否全部合理)
Validity           : 0.XXXX  (RDKit能否成功解析)
Uniqueness         : 0.XXXX  (去重后的比例)
Novelty            : 0.XXXX  (训练集中未见过的比例)
==================================================
```

## 架构设计

### 数据流

```
Input Molecule (coords, atomics, padding_mask)
    ↓
[DINO Encoder] (3D-GSRD RelaTransEncoder, 预训练冻结)
    ↓
Node-level features (B, N, hidden_dim)
    ↓
[VAE Bottleneck] (per-node reparameterization)
    ↓
Latent z (B, N, kl_dim=6)
    ↓
[Transformer Decoder] (可训练)
    ↓
Reconstructed (coords, atomics)
```

### 关键设计

1. **Dense ↔ PyG转换**: `dense_to_pyg_batch()` 将Tabasco的dense tensor转成3D-GSRD需要的PyG稀疏图
2. **全连接图**: encoder输入是全连接图（所有原子对都有边）
3. **Edge特征**: 使用全零占位符，让模型纯靠距离特征学习
4. **VAE bottleneck**: per-node KL散度，权重1e-6
5. **位置编码**: decoder使用sinusoidal PE（参考MOL-AE）

## 文件清单

```
code/tabasco/src/
├── tabasco/models/dino_encoder.py      # 核心适配模块
├── train_dino_ae.py                    # AE训练脚本
├── train_dino_flow.py                  # Flow训练脚本
├── eval_dino_gen.py                    # 生成评估脚本
├── train_dino_ae_stage1_pbs.sh         # Stage 1 PBS作业
├── train_dino_ae_stage2_pbs.sh         # Stage 2 PBS作业
└── train_dino_flow_pbs.sh              # Stage 3 PBS作业
```

## 超参数配置

### Autoencoder (Stage 1/2)
- `gsrd_hidden_dim`: 256 (DINO encoder维度)
- `hidden_dim`: 256 (decoder维度)
- `kl_dim`: 6 (latent维度)
- `num_heads`: 8
- `num_layers`: 8 (decoder层数)
- `batch_size`: 64 (Stage 1), 32 (Stage 2)
- `lr`: 1e-4 (Stage 1), 5e-5 (Stage 2)

### Flow Matching (Stage 3)
- `hidden_dim`: 256
- `num_heads`: 8
- `num_layers`: 6
- `batch_size`: 128
- `lr`: 1e-4

## 预期结果

### Baseline (UniGEM on QM9)
参考UniGEM论文Table 1的QM9结果：
- Atom Stability: ~0.95-0.98
- Molecule Stability: ~0.85-0.90
- Validity: ~0.90-0.95

### DINO版本预期
- **Atom Stability**: 应该≥baseline（预训练encoder理解3D几何和化学价）
- **Molecule Stability**: 应该≥baseline（整体分子结构更合理）
- **Validity**: 应该≥baseline（生成的分子更符合化学规则）
- **Uniqueness**: 应该≥0.95（多样性）
- **Novelty**: 应该≥0.90（泛化能力）

## 故障排查

### 常见问题

1. **维度不匹配**
   - 检查`make_gsrd_args()`中的`node_dim`是否与预训练时一致
   - 检查`edge_dim=1`是否匹配

2. **Checkpoint加载失败**
   - 确认`gsrd_checkpoint`路径正确
   - 检查key前缀是否为`model._orig_mod.encoder.*`

3. **OOM (Out of Memory)**
   - 降低`batch_size`
   - 减少`num_layers`

4. **训练不收敛**
   - 检查`kl_weight`是否过大（默认1e-6）
   - 检查学习率是否合适
   - 查看`coord_loss`和`atom_loss`是否都在下降

## 下一步

完成实验后：
1. 整理结果表格（validity/uniqueness/novelty）
2. 与Tabasco baseline对比
3. 可视化生成的分子（使用RDKit）
4. 分析失败案例（invalid molecules）
5. 更新memory文件记录实验结果
