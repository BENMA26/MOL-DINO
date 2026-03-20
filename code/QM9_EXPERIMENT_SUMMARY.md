# QM9生成实验总结

## 实验设计变更

根据你的建议，实验已从CrossDocked数据集改为**QM9数据集**，以便与UniGEM的benchmark直接对比。

## 关键变更

### 1. 数据集
- **原计划**: CrossDocked (药物分子)
- **现方案**: QM9 (小分子，5种原子类型: H, C, N, O, F)
- **原因**: UniGEM在QM9上有完整的benchmark结果

### 2. 评估指标
使用UniGEM的完整评估体系：
- **Atom Stability**: 每个原子的化学价是否合理
- **Molecule Stability**: 整个分子的所有原子化学价是否都合理
- **Validity**: RDKit能否成功解析并sanitize
- **Uniqueness**: 生成分子去重后的比例
- **Novelty**: 训练集中未见过的分子比例

### 3. 评估实现
- 直接调用UniGEM的`qm9/analyze.py`中的`check_stability()`函数
- 使用与UniGEM相同的bond order判断规则
- 输出格式与UniGEM论文Table 1对齐

## 文件清单

### 核心代码
```
code/tabasco/src/
├── tabasco/models/dino_encoder.py      # DINO encoder适配层
├── train_dino_ae.py                    # AE训练（两阶段）
├── train_dino_flow.py                  # Flow matching训练
├── eval_dino_gen.py                    # 生成评估（UniGEM指标）
└── submit_all_jobs.sh                  # 一键提交脚本
```

### PBS作业脚本
```
code/tabasco/src/
├── train_dino_ae_stage1_pbs.sh         # Stage 1: 冻结encoder
├── train_dino_ae_stage2_pbs.sh         # Stage 2: 解冻encoder
└── train_dino_flow_pbs.sh              # Stage 3: Flow matching
```

### 文档
```
code/
├── EXPERIMENT_GUIDE.md                 # 完整实验指南
└── QM9_EXPERIMENT_SUMMARY.md           # 本文档
```

## 实验流程

### Stage 1: 冻结Encoder训练Decoder (100 epochs)
```bash
qsub train_dino_ae_stage1_pbs.sh
```
- 数据: `/scratch/yuxuan.ren/maben/data/qm9/qm9_train.pt`
- 输出: `/scratch/yuxuan.ren/maben/outputs/dino_ae_qm9_stage1/best.ckpt`

### Stage 2: 解冻Encoder端到端微调 (50 epochs)
```bash
qsub train_dino_ae_stage2_pbs.sh  # 自动依赖Stage 1
```
- 从Stage 1 checkpoint恢复
- 输出: `/scratch/yuxuan.ren/maben/outputs/dino_ae_qm9_stage2/best.ckpt`

### Stage 3: Flow Matching训练 (200 epochs)
```bash
qsub train_dino_flow_pbs.sh  # 自动依赖Stage 2
```
- 在latent space训练flow matching
- 输出: `/scratch/yuxuan.ren/maben/outputs/dino_flow_qm9/best.ckpt`

### Stage 4: 生成评估
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

## 一键提交

```bash
cd /scratch/yuxuan.ren/maben/code/tabasco/src
bash submit_all_jobs.sh
```

这会自动提交3个作业，并设置依赖关系（Stage 2等待Stage 1，Stage 3等待Stage 2）。

## 预期结果

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

## 技术细节

### 架构设计
```
Input (coords, atomics, padding_mask)
    ↓
[DINO Encoder] (3D-GSRD RelaTransEncoder, 预训练冻结)
    ↓ PyG sparse graph
Node features (B, N, 256)
    ↓
[VAE Bottleneck] (per-node, kl_dim=6)
    ↓
Latent z (B, N, 6)
    ↓
[Transformer Decoder] (8 layers, 8 heads)
    ↓
Output (coords, atomics)
```

### 关键设计
1. **Dense ↔ PyG转换**: `dense_to_pyg_batch()`自动转换
2. **全连接图**: encoder输入全连接图（所有原子对）
3. **Edge特征**: 全零占位符，纯靠距离学习
4. **位置编码**: decoder使用sinusoidal PE
5. **两阶段训练**: 先冻结encoder让decoder适配，再端到端微调

### 超参数
- `gsrd_hidden_dim`: 256 (DINO encoder)
- `hidden_dim`: 256 (decoder)
- `kl_dim`: 6 (latent bottleneck)
- `num_heads`: 8
- `num_layers`: 8 (decoder), 6 (flow)
- `batch_size`: 64 (Stage 1), 32 (Stage 2), 128 (Flow)
- `lr`: 1e-4 (Stage 1), 5e-5 (Stage 2), 1e-4 (Flow)

## 下一步

1. **提交作业**: `bash submit_all_jobs.sh`
2. **监控训练**: `qstat -u $USER` 和查看日志文件
3. **运行评估**: 等待所有作业完成后运行`eval_dino_gen.py`
4. **结果对比**: 与UniGEM Table 1对比
5. **分析**: 可视化生成的分子，分析失败案例

## 故障排查

### 数据路径
确认QM9数据集路径：
- Train: `/scratch/yuxuan.ren/maben/data/qm9/qm9_train.pt`
- Val: `/scratch/yuxuan.ren/maben/data/qm9/qm9_val.pt`

### Checkpoint路径
确认DINO预训练checkpoint：
- `/scratch/yuxuan.ren/maben/checkpoints/pretrain.ckpt`

### 维度匹配
- `node_dim=25` (22 atom types + 3 coords)
- `edge_dim=1` (distance only)
- 需与预训练时一致

### UniGEM依赖
评估脚本需要访问UniGEM代码：
- 路径: `../../UniGEM`
- 需要: `qm9/analyze.py`, `qm9/bond_analyze.py`
