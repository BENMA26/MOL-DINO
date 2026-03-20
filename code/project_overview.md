# 项目概览：3D-GSRD 与 TABASCO

## 1. 3D-GSRD

**论文：** NeurIPS 2025
**全称：** 3D Molecular Graph Auto-Encoder with Selective Re-mask Decoding

### 核心任务

分子性质预测（QM9、MD17）和分子生成。通过在大规模数据集（PCQM4Mv2，约 330 万分子）上预训练，再迁移到下游任务。

### 架构

两阶段设计：

**编码器（RelaTransEncoder）**
- 等变图关系 Transformer，处理 3D 分子结构
- 用高斯基函数（GaussianLayer）和指数正态 Smearing（ExpNormalSmearing）展开原子间距离
- 通过 DMTBlock（扩散消息 Transformer 块）提取特征，支持时间条件调制
- 使用等变层归一化（EquivariantLayerNorm）保持旋转等变性

**解码器（StructureUnawareDecoder）**
- 对被掩码的原子位置进行重建，输出坐标和去噪预测
- 核心创新：**选择性重掩码（Selective Re-mask Decoding）**——解码时只对部分位置重新掩码，而非全部
- 使用正弦位置编码 + DecoderBlock 等变注意力层
- 输出头：EquivariantVectorOutput（坐标、偶极矩等）

### 训练流程

1. **预训练**（`run_pretrain.sh`）
   - 数据集：PCQM4Mv2
   - 掩码比例：25%
   - 最大步数：400 万步，余弦退火
   - 局部-全局蒸馏 + 原型对比学习

2. **微调**
   - QM9：分子性质预测（HOMO、LUMO、偶极矩等）
   - MD17：分子动力学力场预测

### 关键文件

| 文件 | 说明 |
|------|------|
| `model/autoencoder.py` | 主模型（AutoEncoder） |
| `model/retrans.py` | 编码器（RelaTransEncoder） |
| `model/output_modules.py` | 等变输出头 |
| `data_provider/` | QM9、MD17、PCQM4Mv2 数据集处理 |
| `trainer_qm9.py` | QM9 微调入口 |
| `trainer_md17.py` | MD17 微调入口 |
| `trainer_pcqm4mv2.py` | 预训练入口 |
| `atomref.py` | 原子参考能量先验 |
| `training_utils.py` | 回调、检查点、设备管理 |

### 依赖

- PyTorch 2.4.1 + CUDA 12.1
- PyTorch Geometric
- PyTorch Lightning
- RDKit、OpenBabel

---

## 2. TABASCO

**论文：** GenBio Workshop @ ICML 2025
**全称：** A Fast, Simplified Model for Molecular Generation with Improved Physical Quality

### 核心任务

无条件分子生成，目标是在保持化学合理性（PoseBusters 评分）的同时实现 **10x 采样加速**。

### 架构

**流匹配模型（FlowMatchingModel）**
- 用两个插值器分别处理不同模态：
  - `SDEMetricInterpolant`：连续坐标流
  - `DiscreteInterpolant`：离散原子类型（交叉熵损失）
- 时间采样：Beta 分布（alpha=1.8），非均匀采样
- 支持随机旋转增强（每样本 7 次旋转）
- 采样：显式欧拉步进

**Transformer 网络（TransformerModule）**
- 标准**非等变** Transformer，把分子生成当作序列建模
- 16 层，8 头，隐层维度 256，激活函数 SiLU
- 支持坐标域与原子类型域之间的交叉注意力
- 正弦位置编码 + 时间傅里叶编码
- 不依赖等变性，因此推理更快

**采样方式**
- 无条件采样：100 步扩散（`src/sample.py`）
- 引导采样：UFF 约束引导，提升物理合理性（`src/sample_uff_bounds.py`）

### 训练流程

- 数据集：GEOM-Drugs、QM9
- 数据存储：LMDB 高效批处理
- Batch size：256，Workers：31
- 学习率：0.002，EMA decay：0.999
- 配置系统：Hydra（`configs/experiment/hot_geom.yaml` 等）
- 日志：Weights & Biases

### 关键文件

| 文件 | 说明 |
|------|------|
| `src/train.py` | 训练入口（Hydra 配置） |
| `src/sample.py` | 无条件采样 |
| `src/sample_uff_bounds.py` | UFF 约束引导采样 |
| `models/flow_model.py` | 核心流匹配模型 |
| `models/lightning_tabasco.py` | Lightning 封装 + 评估指标 |
| `models/components/transformer_module.py` | Transformer 实现 |
| `models/components/transformer.py` | 标准 Transformer 块 |
| `models/components/positional_encoder.py` | 位置编码 |
| `flow/interpolate.py` | 插值器基类与实现 |
| `flow/path.py` | FlowPath 数据结构 |
| `chem/convert.py` | RDKit ↔ TensorDict 转换 |
| `data/lmdb_datamodule.py` | LMDB 数据模块 |
| `configs/` | Hydra 配置文件 |

### 依赖

- Python 3.11，PyTorch 2.5.1
- PyTorch Lightning 2.*
- Hydra 1.3.*
- RDKit 2024.09.4，Datamol 0.12.*，OpenBabel 3.1
- LMDB 1.*，TensorDict 0.7.0
- PoseBusters 0.3.1
- Weights & Biases

---

## 两者对比

| 维度 | 3D-GSRD | TABASCO |
|------|---------|---------|
| 任务 | 性质预测 + 生成 | 分子生成 |
| 网络类型 | 等变图 Transformer | 非等变序列 Transformer |
| 生成范式 | 离散掩码 + 去噪 | 连续流匹配 |
| 采样速度 | 标准 | 快 10x |
| 参数效率 | 较大（等变操作开销） | 更高效 |
| 训练策略 | 预训练 + 微调 | 端到端训练 |
| 数据集 | QM9 / MD17 / PCQM4Mv2 | GEOM-Drugs / QM9 |
| 配置系统 | Shell 脚本 | Hydra |

---

## 总结

- **3D-GSRD** 重视表达能力，使用等变图网络 + 选择性重掩码解码，适合需要精确性质预测的场景，通过大规模预训练获得强泛化能力。
- **TABASCO** 以速度为优先，放弃等变性，用流匹配 + 标准 Transformer 实现高效分子生成，通过 UFF 约束引导保证化学合理性。
- 两者都以 QM9 为基准，可作为图 tokenizer 相关工作的上游基础模型。

---

## 修改记录（基于 git 历史）

作者：**YuX-Ren**（yuxuanren@ustc.edu）

### 3D-GSRD 的修改

原始代码（`fba4015 Add files via upload`）基础上，做了以下改动：

| 提交 | 时间 | 改动内容 |
|------|------|----------|
| `add constrastive learning` | 最新 | 在 `model/retrans.py` 中加入对比学习模块；大幅重构 `trainer_pcqm4mv2.py`（+245/-65行），加入对比损失训练逻辑；更新预训练脚本 |
| `remove 2d part and add postion embed` | 早期 | 删除 2D 图相关部分；在 `model/autoencoder.py` 中加入位置编码；重构 `data_provider/pcqm4mv2.py` 数据处理 |
| 若干 `update` 提交 | 早期 | 调整训练脚本参数（`run_pretrain.sh`、`ft_md17.sh`）；添加 `.gitignore` |

**核心改动方向：** 在原始自编码器预训练框架上叠加了**对比学习（Contrastive Learning）**，并移除了 2D 图部分，专注于 3D 结构 + 位置编码。

---

### TABASCO 的修改

原始代码（`feb8d6a Initial commit`）基础上，经历了大量扩展，按时间顺序：

#### 阶段一：基础修复与评估（2025年10月）

| 提交 | 改动内容 |
|------|----------|
| `update yaml` | 添加 QM9 实验配置；调整 wandb/trainer 参数 |
| `sampler rmsd and atom type comparison` | 在 `src/sample.py` 和 `chem/convert.py` 中加入 RMSD 计算与原子类型对比评估 |
| `remove SDE sampling` | 简化采样流程，移除 SDE 采样 |
| `add pos embedding to codes` | 在 Transformer 的 codebook 中加入位置编码 |
| `add rmsd_caculations` | 大幅扩展采样脚本，加入完整 RMSD 评估流程 |
| `add diffusion autoencoders` | **关键改动**：加入 FSQ（Finite Scalar Quantization）和 SimVQ 两种向量量化模块（`fsq.py`、`simvq.py`），将 TABASCO 改造为支持 tokenizer/codebook 的扩散自编码器 |

#### 阶段二：扩展到晶体材料（2025年10月-11月）

| 提交 | 改动内容 |
|------|----------|
| `add materials` | 加入晶体材料数据集支持（MP-20/QMOF）；新增 `lmdb_unconditional_crystal.py`；在 `flow/interpolate.py` 中加入晶格插值；`transformer_module.py` 支持晶格特征 |
| `lattices as virtual nodes` | 将晶格参数编码为虚拟节点输入 Transformer；加入晶体结构匹配器 `crystal_matcher.py` |
| `config for materials` | 添加材料专用 Hydra 配置 |
| `joint mol & materials` | 支持分子与晶体材料的联合训练；大幅重构 `lmdb_datamodule.py` |

#### 阶段三：扩展到蛋白质（2025年12月-2026年1月）

| 提交 | 改动内容 |
|------|----------|
| `add protein dataset` | 加入 PDB 蛋白质数据集（`lmdb_pdb.py`）；`transformer_module.py` 支持蛋白质特征（氨基酸类型、骨架坐标） |
| `mol diffusion` | 引入完整的 **LDM（Latent Diffusion Model）** 模块（`ldm_module.py`，720行）；加入 DiT（Diffusion Transformer）去噪器；新增 `train_diffusion.py` 训练入口 |
| `protein diffusion` | 加入蛋白质折叠模型（`folding_model.py`）；扩展采样脚本支持蛋白质生成；加入评估脚本（晶体/MOF/分子/蛋白质重建与生成） |
| `joint diffusion` | 支持分子+材料联合扩散训练；加入 `ase_notebook` 可视化工具；新增多套 KL 系数实验配置（kl4/kl8/kl16/kl32） |
| `in batch mix training` | 支持同一 batch 内混合不同模态数据训练 |
| `add proteina training` | **最大改动**（155文件，+30710行）：引入完整 OpenFold 代码；加入蛋白质 LDM（`proteinae_ldm.py`，985行）；加入 AlphaFold3 风格的 Transformer 工具；加入蛋白质 Transformer（`protein_transformer.py`，1460行）；移除旧的评估脚本和 ase_notebook |

**核心改动方向：** 将 TABASCO 从单一的小分子生成模型，扩展为支持**分子 + 晶体材料 + 蛋白质**的统一生成框架，并引入了 **VAE/VQ-VAE（FSQ/SimVQ）+ LDM** 的两阶段生成范式，即先训练一个图 tokenizer（自编码器），再在隐空间上训练扩散模型。
