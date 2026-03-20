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
├── ft_md17_pbs.sh         # PBS集群微调脚本（已创建）
└── run_pretrain.sh        # 预训练脚本
```

**关键参数**：
- 预训练：`--local_global_distill --pos_mask --denoising`
- 微调：`--checkpoint_path /path/to/pretrain.ckpt`
- load_encoder_params key前缀：`model._orig_mod.encoder.*`

### Tabasco（3D分子生成框架）
```
code/tabasco/src/tabasco/
├── models/
│   ├── flow_model.py      # FlowMatchingModel：encode/decode接口
│   └── ldm_module.py      # LatentDiffusionLitModule：冻结AE训练denoiser
└── ...
```

**工作流**：
1. autoencoder.encode(batch) → latent z
2. flow matching在latent space做插值
3. autoencoder.decode(z) → 分子

### UniGEM（生成评估）
```
code/UniGEM/
├── eval_analyze.py        # 生成评估：stability/validity/uniqueness
└── eval_sample.py         # 采样可视化
```

---

## 三、已完成工作

- [x] 阅读并理解5篇论文（3D-GSRD, DINO, MOL-AE, UniGEM, UniLIP）
- [x] 理解3D-GSRD代码库工作流程
- [x] 理解Tabasco生成框架（autoencoder + flow matching）
- [x] 创建PBS微调脚本 `ft_md17_pbs.sh`
- [x] 成功提交MD17微调job（job 346400）

---

## 四、Todo List

### 主线1：性质预测实验（MD17）

- [ ] **等待job完成**：job 346400，8个分子分别跑
- [ ] **收集结果**：每个分子的力预测MAE（单位：kcal/mol/Å）
- [ ] **对比Table 2**：与3D-GSRD原始结果对比，验证Molecule DINO encoder的性质预测能力
- [ ] **（可选）QM9实验**：如果MD17结果好，进一步在QM9上验证

**预期结果格式**：
| Molecule | Force MAE (ours) | Force MAE (3D-GSRD) |
|----------|-----------------|---------------------|
| Aspirin  | ?               | ?                   |
| ...      | ...             | ...                 |

---

### 主线2：生成任务实验

**目标**：将Molecule DINO encoder接入Tabasco生成框架，验证预训练表征对生成质量的提升

**实验设计**：

#### Stage 1：替换Tabasco的encoder
- 用Molecule DINO预训练的encoder替换Tabasco autoencoder中的encoder部分
- 冻结encoder，只训练decoder（参考UniLIP Stage1）
- 位置编码参考MOL-AE：SMILES顺序sinusoidal PE，只加在decoder

#### Stage 2：端到端微调（可选）
- 解冻encoder，自蒸馏微调（参考UniLIP Stage2）
- teacher=encoder的EMA，保持表征稳定性

#### Stage 3：训练生成模型
- 冻结autoencoder，在latent space训练flow matching（参考Tabasco LDM）
- 评估：QM9上atom stability、mol stability、validity、uniqueness、novelty

**需要实现的代码**：
- [ ] 将3D-GSRD encoder权重加载到Tabasco autoencoder
- [ ] 修改Tabasco decoder，加入MOL-AE风格位置编码
- [ ] 实现两阶段训练逻辑（Stage1冻结encoder）
- [ ] 对接UniGEM评估脚本，计算生成指标

**关键问题待确认**：
- Tabasco encoder的输入格式 vs 3D-GSRD encoder的输入格式是否兼容？
- latent space维度是否需要调整？
- 是否需要重新设计decoder架构？

---

## 五、下次启动时的优先级

1. 检查job 346400的运行状态和结果
2. 如果MD17结果出来了，整理对比表格
3. 开始设计生成实验的代码实现方案
