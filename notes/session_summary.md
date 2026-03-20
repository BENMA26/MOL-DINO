# 会话整合：Molecule DINO 实验设计

## 一、论文速览

### 3D-GSRD
- **问题**：Masked Graph Modeling中，解码器可以通过2D结构（化学键）推断被mask的3D坐标，导致encoder不需要真正学习3D信息
- **方案**：Selective Re-mask Decoding（SRD）——只re-mask 3D坐标，保留2D结构；配合结构无关解码器（只用位置编码，不用GNN）和3D-ReTrans编码器（处理标量+向量特征）
- **优点**：encoder被迫学习真正的3D几何信息；解码器无法作弊
- **缺点**：需要3D坐标输入；位置编码设计较复杂

### DINO
- **问题**：无标签自监督学习，如何避免表征坍塌
- **方案**：自蒸馏，多个student网络共享参数，teacher=student的EMA；local子图→全局语义，cross-entropy loss
- **关键**：teacher网络不反向传播，只做EMA更新；centering防止坍塌
- **最终使用**：teacher网络（更稳定）

### MOL-AE
- **核心设计**：位置编码只加在decoder，基于SMILES顺序的sinusoidal PE
- **意义**：encoder学到的是与顺序无关的全局表征；decoder通过PE知道"要重建哪个位置的原子"

### UniGEM
- **方案**：两阶段扩散生成
  - Nucleation phase：生成分子骨架（原子类型+大致位置）
  - Growth phase：细化3D坐标
- **统一**：同一模型做生成+性质预测
- **生成实验**：QM9上atom stability、mol stability、validity、uniqueness、novelty

### UniLIP
- **问题**：CLIP只有理解能力，没有生成/重建能力
- **方案**：两阶段训练+自蒸馏，双条件架构（文本条件+图像条件）
- **对我们的启发**：Stage1冻结encoder训练decoder，Stage2自蒸馏微调encoder

---

## 二、代码库结构

### 3D-GSRD（Molecule DINO预训练）
```
code/3D-GSRD/
├── trainer_pcqm4mv2.py    # 预训练：DINO local-global蒸馏
├── trainer_md17.py        # 微调：预测E，F=-∇E
├── training_utils.py      # load_encoder_params（key前缀匹配）
├── model/
│   ├── autoencoder.py     # AutoEncoder = encoder + decoder
│   └── retrans.py         # 3D-ReTrans实现
└── ft_md17_pbs.sh         # PBS集群微调脚本
```

**关键参数**：
- 预训练：`--local_global_distill --pos_mask --denoising`
- 微调：`--checkpoint_path /path/to/pretrain.ckpt`
- load_encoder_params key前缀：`model._orig_mod.encoder.*`

### Tabasco（3D分子生成框架）+ DINO集成
```
code/tabasco/src/
├── tabasco/models/
│   ├── dino_encoder.py         # 核心适配层（NEW）
│   ├── flow_model.py           # FlowMatchingModel
│   └── ldm_module.py           # LatentDiffusionLitModule
├── train_dino_ae.py            # AE训练脚本（NEW）
├── train_dino_flow.py          # Flow训练脚本（NEW）
├── eval_dino_gen.py            # 生成评估脚本（NEW）
├── train_dino_ae_stage1_pbs.sh # Stage 1 PBS作业（NEW）
├── train_dino_ae_stage2_pbs.sh # Stage 2 PBS作业（NEW）
├── train_dino_flow_pbs.sh      # Flow训练PBS作业（NEW）
├── submit_all_jobs.sh          # 一键提交脚本（NEW）
└── README_DINO_GENERATION.md   # 完整文档（NEW）
```

**工作流**：
1. DINO encoder: dense → PyG sparse graph → node features
2. VAE bottleneck: node features → latent z (per-node)
3. Transformer decoder: latent z → reconstructed molecule
4. Flow matching: 在latent space生成新分子

### UniGEM（生成评估）
```
code/UniGEM/
├── eval_analyze.py        # 生成评估：stability/validity/uniqueness
├── qm9/analyze.py         # check_stability()函数
└── qm9/bond_analyze.py    # bond order判断规则
```

---

## 三、已完成工作

### 主线1：性质预测实验（MD17）
- [x] 阅读并理解5篇论文（3D-GSRD, DINO, MOL-AE, UniGEM, UniLIP）
- [x] 理解3D-GSRD代码库工作流程
- [x] 创建PBS微调脚本 `ft_md17_pbs.sh`
- [x] 成功提交MD17微调job（job 346400）
- [x] **实验已完成**

### 主线2：生成任务实验（QM9 + UniGEM Benchmark）
- [x] **完整实现所有代码**
  - [x] DinoEncoderModule适配层（`dino_encoder.py`）
    - Dense ↔ PyG转换
    - 加载预训练DINO encoder
    - Per-node VAE bottleneck (kl_dim=6)
    - Transformer decoder (8 layers, 8 heads)
  - [x] 两阶段AE训练脚本（`train_dino_ae.py`）
    - Stage 1: 冻结encoder，训练decoder (100 epochs)
    - Stage 2: 解冻encoder，端到端微调 (50 epochs)
  - [x] Flow matching训练脚本（`train_dino_flow.py`）
    - 在latent space训练flow matching (200 epochs)
    - OT interpolant (coords) + VP interpolant (atomics)
  - [x] 生成评估脚本（`eval_dino_gen.py`）
    - 集成UniGEM的check_stability()函数
    - 计算atom/mol stability, validity, uniqueness, novelty
  - [x] PBS作业脚本（3个stage）
    - `train_dino_ae_stage1_pbs.sh`
    - `train_dino_ae_stage2_pbs.sh`
    - `train_dino_flow_pbs.sh`
  - [x] 一键提交脚本（`submit_all_jobs.sh`）
    - 自动设置作业依赖关系

- [x] **完整文档**
  - [x] `README_DINO_GENERATION.md` - 完整技术文档
  - [x] `EXPERIMENT_GUIDE.md` - 详细实验指南
  - [x] `QM9_EXPERIMENT_SUMMARY.md` - QM9实验总结
  - [x] `QUICKSTART.md` - 快速启动指南
  - [x] `README.md` - 项目主README

- [x] **GitHub仓库**
  - [x] 清除所有子模块的.git信息
  - [x] 初始化新的git仓库
  - [x] 提交所有代码（281个文件，204,877行）
  - [x] 推送到GitHub: https://github.com/BENMA26/MOL-DINO.git

---

## 四、实验设计详解

### 数据集变更
- **原计划**: CrossDocked (药物分子)
- **最终方案**: QM9 (小分子，5种原子类型: H, C, N, O, F)
- **原因**: UniGEM在QM9上有完整的benchmark结果，便于直接对比

### 评估指标（UniGEM Benchmark）
1. **Atom Stability**: 每个原子的化学价是否合理（基于bond order）
2. **Molecule Stability**: 整个分子的所有原子化学价是否都合理
3. **Validity**: RDKit能否成功解析并sanitize分子
4. **Uniqueness**: 生成分子去重后的比例
5. **Novelty**: 训练集中未见过的分子比例

### 三阶段训练流程

**Stage 1: 冻结Encoder训练Decoder (100 epochs)**
- 目的: 让decoder学会从DINO encoder的表征重建分子
- 超参数: batch_size=64, lr=1e-4, freeze_encoder=True
- 输出: `outputs/dino_ae_qm9_stage1/best.ckpt`

**Stage 2: 解冻Encoder端到端微调 (50 epochs)**
- 目的: 微调encoder使其更适配生成任务
- 超参数: batch_size=32, lr=5e-5, freeze_encoder=False
- 从Stage 1 checkpoint恢复
- 输出: `outputs/dino_ae_qm9_stage2/best.ckpt`

**Stage 3: Flow Matching训练 (200 epochs)**
- 目的: 在latent space学习分子分布
- 超参数: batch_size=128, lr=1e-4
- 冻结autoencoder
- 输出: `outputs/dino_flow_qm9/best.ckpt`

### 架构设计

```
Input Molecule (coords, atomics, padding_mask)
    ↓
[dense_to_pyg_batch] ← Dense (B,N,F) → PyG Batch (全连接图)
    ↓
[DINO Encoder] ← 3D-GSRD RelaTransEncoder (预训练冻结)
    ↓
Node Features (B, N, 256)
    ↓
[VAE Bottleneck] ← per-node reparameterization
    ↓ μ, σ → z = μ + σ * ε
Latent z (B, N, 6)
    ↓
[Transformer Decoder] ← 8 layers, 8 heads, sinusoidal PE
    ↓
Reconstructed (coords, atomics)
    ↓
[Flow Matching] ← OT + VP interpolants
    ↓
Generated Molecule
```

### 关键技术细节

1. **Dense ↔ PyG转换**
   - `dense_to_pyg_batch()`: 将(B,N,3)坐标和(B,N,A)原子类型转成PyG Batch
   - 全连接图: 所有原子对都有边
   - Edge特征: 全零占位符，纯靠距离学习

2. **VAE Bottleneck**
   - Per-node reparameterization: (B, N, 256) → (B, N, 6)
   - KL weight: 1e-6
   - 训练时采样，推理时使用均值

3. **位置编码**
   - 只加在decoder，不加在encoder
   - Sinusoidal PE（参考MOL-AE）
   - Encoder学到与顺序无关的全局表征

---

## 五、预期结果

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

---

## 六、下一步行动

### 立即执行
1. **提交实验作业**
   ```bash
   cd /scratch/yuxuan.ren/maben/code/tabasco/src
   bash submit_all_jobs.sh
   ```

2. **监控作业状态**
   ```bash
   qstat -u $USER
   tail -f dino_ae_stage1.log
   ```

### 等待完成后
3. **运行评估**
   ```bash
   python eval_dino_gen.py \
       --ae_ckpt outputs/dino_ae_qm9_stage2/best.ckpt \
       --flow_ckpt outputs/dino_flow_qm9/best.ckpt \
       --ref_data_dir /path/to/qm9_train.pt \
       --n_samples 1000 \
       --num_steps 100 \
       --batch_size 64
   ```

4. **结果分析**
   - 与UniGEM baseline对比
   - 分析失败案例
   - 可视化生成的分子

---

## 七、项目资源

### GitHub仓库
- **地址**: https://github.com/BENMA26/MOL-DINO.git
- **内容**: 完整代码库 + 文档 + 论文材料
- **文件数**: 281个文件，204,877行代码

### 文档索引
- `README.md` - 项目主README
- `code/tabasco/src/README_DINO_GENERATION.md` - 完整技术文档
- `code/EXPERIMENT_GUIDE.md` - 详细实验指南
- `code/QM9_EXPERIMENT_SUMMARY.md` - QM9实验总结
- `code/QUICKSTART.md` - 快速启动指南

### 关键文件路径
- 核心代码: `code/tabasco/src/tabasco/models/dino_encoder.py`
- 训练脚本: `code/tabasco/src/train_dino_*.py`
- PBS脚本: `code/tabasco/src/*_pbs.sh`
- 一键提交: `code/tabasco/src/submit_all_jobs.sh`

---

## 八、技术亮点

1. **预训练表征**: 利用DINO自蒸馏学到的3D几何表征
2. **两阶段训练**: 先适配后微调，保持预训练知识
3. **Flow Matching**: 在latent space生成，效率更高
4. **严格评估**: 使用UniGEM的stability指标，不只看validity
5. **完整实现**: 从数据处理到评估的完整pipeline

---

**最后更新**: 2026-03-21

**状态**: ✅ 代码实现完成，✅ 文档完成，✅ GitHub上传完成，⏳ 等待实验结果
