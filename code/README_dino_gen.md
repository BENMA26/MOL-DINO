# DINO Encoder + Flow Matching 生成实验

## 新增文件

```
tabasco/src/
├── tabasco/models/dino_encoder.py   # 核心适配层
├── train_dino_ae.py                 # AE训练脚本（Stage 1/2）
├── eval_dino_gen.py                 # 生成评估脚本
├── train_dino_ae_stage1_pbs.sh      # PBS Stage 1 作业
└── train_dino_ae_stage2_pbs.sh      # PBS Stage 2 作业
```

## 架构设计

### 接口适配（`dino_encoder.py`）

3D-GSRD encoder 输入是 PyG 稀疏图，Tabasco 输入是 dense tensor，需要适配：

- `dense_to_pyg_batch()`：把 `(B,N,3)` 坐标 + `(B,N,A)` 原子类型转成全连接 PyG Batch
- encoder 输出 node-level 特征，用 `to_dense_batch` 还原成 `(B,N,H)`
- `NodeVAEBottleneck`：per-node VAE reparameterization（和 Tabasco 原版一致）
- decoder：完全复用 Tabasco 的 Transformer decoder 结构

### 两阶段训练

**Stage 1**：冻结 DINO encoder，只训练 decoder + bottleneck
```bash
python train_dino_ae.py \
    --gsrd_checkpoint /path/to/pretrain.ckpt \
    --data_dir /path/to/train.pt \
    --val_data_dir /path/to/val.pt \
    --output_dir ./outputs/dino_ae_stage1 \
    --freeze_encoder \
    --max_epochs 100 \
    --batch_size 64 \
    --lr 1e-4
```

**Stage 2**：解冻 encoder，从 Stage 1 checkpoint 恢复，全参数微调
```bash
python train_dino_ae.py \
    --gsrd_checkpoint /path/to/pretrain.ckpt \
    --data_dir /path/to/train.pt \
    --val_data_dir /path/to/val.pt \
    --output_dir ./outputs/dino_ae_stage2 \
    --resume_from ./outputs/dino_ae_stage1/best.ckpt \
    --no_freeze_encoder \
    --max_epochs 50 \
    --batch_size 32 \
    --lr 5e-5
```

### 生成评估

```bash
python eval_dino_gen.py \
    --ae_ckpt ./outputs/dino_ae_stage2/best.ckpt \
    --flow_ckpt ./outputs/dino_flow/best.ckpt \
    --ref_data_dir /path/to/train.pt \
    --n_samples 1000 \
    --num_steps 100 \
    --batch_size 64
```

输出指标：Validity、Uniqueness、Novelty

### PBS 集群

```bash
qsub train_dino_ae_stage1_pbs.sh   # Stage 1
qsub train_dino_ae_stage2_pbs.sh   # Stage 2（Stage 1 完成后）
```

## 运行前需要确认的事项

1. **node_dim 对齐**：`make_gsrd_args()` 里 `node_dim=22+3`，需要和预训练时的 featurization 一致（检查 `featurization.py` 里 `merge_types` 的维度）

2. **edge_attr 维度**：`gsrd_args.edge_dim=1`，encoder 使用全零占位符（纯距离特征），需确认和预训练 encoder 构造时的 `edge_dim` 匹配

3. **数据路径**：PBS 脚本里的路径改成集群上的实际路径

4. **flow matching 训练**：AE 训练完成后，还需要用 Tabasco 原有的 `train_diffusion.py` 在 latent space 训练 flow matching 模型，再用 `eval_dino_gen.py` 联合评估
