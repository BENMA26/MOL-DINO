# 🚀 Quick Start Guide

DINO Encoder + Tabasco生成实验快速启动指南

## ⚡ 一键启动

```bash
cd /scratch/yuxuan.ren/maben/code/tabasco/src
bash submit_all_jobs.sh
```

这会自动提交3个作业（Stage 1 → Stage 2 → Stage 3），并设置依赖关系。

## 📊 监控进度

```bash
# 查看作业状态
qstat -u $USER

# 实时查看日志
tail -f dino_ae_stage1.log
tail -f dino_ae_stage2.log
tail -f dino_flow.log
```

## 🎯 三阶段流程

### Stage 1: 冻结Encoder (100 epochs)
- **作业**: `train_dino_ae_stage1_pbs.sh`
- **输出**: `outputs/dino_ae_qm9_stage1/best.ckpt`
- **时间**: ~24-48小时

### Stage 2: 解冻Encoder (50 epochs)
- **作业**: `train_dino_ae_stage2_pbs.sh`
- **输出**: `outputs/dino_ae_qm9_stage2/best.ckpt`
- **时间**: ~12-24小时

### Stage 3: Flow Matching (200 epochs)
- **作业**: `train_dino_flow_pbs.sh`
- **输出**: `outputs/dino_flow_qm9/best.ckpt`
- **时间**: ~48-72小时

## 🧪 评估生成质量

等待所有作业完成后：

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

## 📈 预期结果

### UniGEM Baseline (QM9)
- Atom Stability: ~0.95-0.98
- Molecule Stability: ~0.85-0.90
- Validity: ~0.90-0.95

### DINO目标
- Atom Stability: ≥0.96
- Molecule Stability: ≥0.88
- Validity: ≥0.92
- Uniqueness: ≥0.95
- Novelty: ≥0.90

## 🔧 手动提交（可选）

如果不想用一键脚本，可以手动提交：

```bash
# Stage 1
JOB1=$(qsub train_dino_ae_stage1_pbs.sh)

# Stage 2 (等待Stage 1完成)
JOB2=$(qsub -W depend=afterok:$JOB1 train_dino_ae_stage2_pbs.sh)

# Stage 3 (等待Stage 2完成)
JOB3=$(qsub -W depend=afterok:$JOB2 train_dino_flow_pbs.sh)
```

## 📁 输出文件位置

```
/scratch/yuxuan.ren/maben/outputs/
├── dino_ae_qm9_stage1/
│   ├── best.ckpt
│   └── logs/
├── dino_ae_qm9_stage2/
│   ├── best.ckpt
│   └── logs/
└── dino_flow_qm9/
    ├── best.ckpt
    └── logs/
```

## ⚠️ 前置检查

提交作业前确认：

```bash
# 1. 数据集存在
ls /scratch/yuxuan.ren/maben/data/qm9/qm9_train.pt
ls /scratch/yuxuan.ren/maben/data/qm9/qm9_val.pt

# 2. DINO checkpoint存在
ls /scratch/yuxuan.ren/maben/checkpoints/pretrain.ckpt

# 3. 环境激活
conda activate mol_dino

# 4. 在正确目录
cd /scratch/yuxuan.ren/maben/code/tabasco/src
```

## 🐛 常见问题

**作业失败？**
```bash
# 查看错误日志
cat dino_ae_stage1.log | grep -i error
```

**OOM错误？**
```bash
# 降低batch_size
# 编辑PBS脚本，修改 --batch_size 参数
```

**Checkpoint加载失败？**
```bash
# 检查路径是否正确
ls -lh /scratch/yuxuan.ren/maben/checkpoints/pretrain.ckpt
```

## 📚 详细文档

- `README_DINO_GENERATION.md`: 完整README
- `EXPERIMENT_GUIDE.md`: 详细实验指南
- `QM9_EXPERIMENT_SUMMARY.md`: QM9实验总结

## ✅ Checklist

- [ ] 数据集准备完成
- [ ] DINO checkpoint准备完成
- [ ] 环境激活 (mol_dino)
- [ ] 提交作业 (`bash submit_all_jobs.sh`)
- [ ] 监控作业状态 (`qstat -u $USER`)
- [ ] 等待所有作业完成
- [ ] 运行评估脚本
- [ ] 对比UniGEM baseline
- [ ] 记录实验结果

---

**预计总时间**: 3-5天（取决于集群负载）

**下一步**: 提交作业后，定期检查日志文件，确保训练正常进行。
