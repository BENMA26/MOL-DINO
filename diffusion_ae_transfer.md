# Flow Matching Autoencoder 迁移指南

## 背景与目标

将两个独立代码仓库合并，构建一个新的分子生成模型：

- **编码器仓库**（`3D-GSRD`）：基于 `RelaTransEncoder` 的 GNN 编码器，使用空间近邻消息传递
- **解码器仓库**（`tabasco`）：基于 `TransformerModule` 的 Flow Matching 解码器，使用全局自注意力

**目标架构**：
```
PyG batch
  └─ RelaTransEncoder         # 编码器：局部消息传递，提取 per-atom 表示
       └─ to_dense_batch      # 格式转换：打包格式 → padding 格式
            └─ VAE 瓶颈       # quant_conv: [B,N,H] → [B,N,kl_dim]，重参数化
                 └─ DAE 解码器 # quant_conv_out + TransformerModule.decode_z
                      └─ FlowMatchingModel  # 流匹配训练/推理框架
```

---

## 路径约定

```
ENCODER_ROOT = /Users/benma/Desktop/projects/graph_tokenizer/diffusion decoder/tabasco/3D-GSRD
DECODER_ROOT = /Users/benma/Desktop/projects/graph_tokenizer/diffusion decoder/tabasco/tabasco
```

**所有新建文件都放在 `ENCODER_ROOT` 下**，以编码器仓库为主代码库。

---

## 第一步：阅读源码，建立理解

在开始任何修改之前，请先阅读以下文件，理解关键接口：

### 1.1 阅读编码器

```
ENCODER_ROOT/
  ├── models/encoder.py 或对应文件   # RelaTransEncoder 的完整定义
  ├── trainer/autoencoder_trainer.py  # AutoEncoderTrainer，理解数据流
  └── data/dataset.py 或对应文件     # 数据格式，确认 data.x / data.pos / data.edge_index 的维度
```

需要确认：
- `RelaTransEncoder.forward` 的完整签名
- `data.x` 的形状和含义（原子类型的编码方式，是 one-hot 还是整数 index）
- `data.edge_attr` 是否存在，是否为 None（coauthor 说去掉了 2D 输入）
- `args.hidden_dim`、`args.node_dim`、`args.edge_dim`、`args.n_heads`、`args.encoder_blocks` 的默认值

### 1.2 阅读解码器

```
DECODER_ROOT/
  ├── tabasco/models/components/transformer_module.py  # TransformerModule 完整定义
  ├── tabasco/flow/interpolate.py                      # Interpolant
  ├── tabasco/flow/path.py                             # FlowPath
  ├── tabasco/flow/utils.py                            # HistogramTimeDistribution
  ├── tabasco/models/flow_matching_model.py            # FlowMatchingModel
  ├── tabasco/models/components/losses.py              # InterDistancesLoss, compute_rmsd_with_kabsch
  └── tabasco/data/transforms.py                       # apply_random_rotation
```

需要确认：
- `TransformerModule.decode_z` 的完整签名和内部依赖的所有子模块
- `FlowMatchingModel.__init__` 需要哪些参数
- `Interpolant` 的构造方式
- `TransformerModule` 中 `decode_z` 使用了哪些 `self.*` 属性（需要全部迁移）

---

## 第二步：在编码器仓库中创建 flow 目录，迁移解码器组件

### 2.1 创建目录结构

在 `ENCODER_ROOT` 下新建以下目录和空的 `__init__.py`：

```
ENCODER_ROOT/
  └── flow_matching/
      ├── __init__.py
      ├── flow/
      │   ├── __init__.py
      │   ├── interpolate.py
      │   ├── path.py
      │   └── utils.py
      ├── models/
      │   ├── __init__.py
      │   ├── flow_matching_model.py
      │   ├── combined_net.py          ← 新建
      │   └── components/
      │       ├── __init__.py
      │       └── losses.py
      └── data/
          ├── __init__.py
          └── transforms.py
```

### 2.2 直接复制的文件（不需要修改）

将以下文件从解码器仓库**原样复制**到编码器仓库对应位置：

| 源文件（DECODER_ROOT） | 目标位置（ENCODER_ROOT） |
|---|---|
| `tabasco/flow/interpolate.py` | `flow_matching/flow/interpolate.py` |
| `tabasco/flow/path.py` | `flow_matching/flow/path.py` |
| `tabasco/flow/utils.py` | `flow_matching/flow/utils.py` |
| `tabasco/models/flow_matching_model.py` | `flow_matching/models/flow_matching_model.py` |
| `tabasco/models/components/losses.py` | `flow_matching/models/components/losses.py` |
| `tabasco/data/transforms.py` | `flow_matching/data/transforms.py` |

复制后检查每个文件的 import 路径，将所有 `from tabasco.xxx` 替换为 `from flow_matching.xxx`。

### 2.3 提取解码器组件

从 `DECODER_ROOT/tabasco/models/components/transformer_module.py` 中提取 `decode_z` 依赖的所有组件，新建文件 `ENCODER_ROOT/flow_matching/models/components/dae_decoder.py`。

需要提取的内容：
- `SinusoidEncoding` 类
- `TimeFourierEncoding` 类
- `SwiGLU` 类（如果存在）
- `Transformer` 类（如果 `implementation=="reimplemented"` 用到）
- `DAEDecoder` 类（见下方说明）

**`DAEDecoder` 的构造**：不要复制整个 `TransformerModule`，只提取解码相关部分，新建 `DAEDecoder`：

```python
class DAEDecoder(nn.Module):
    """
    从 TransformerModule 中提取的解码器部分。
    只保留 decode_z 需要的组件，去掉编码器相关的所有内容。
    """
    def __init__(
        self,
        spatial_dim: int,
        atom_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,          # 对应原来的 num_layers（完整层数，不除以4）
        activation: str = "SiLU",
        add_sinusoid_posenc: bool = True,
        concat_combine_input: bool = False,
        cross_attention: bool = False,
        implementation: str = "pytorch",
    ):
        super().__init__()
        # 从 TransformerModule.__init__ 中复制以下属性：
        self.hidden_dim = hidden_dim
        self.add_sinusoid_posenc = add_sinusoid_posenc
        self.concat_combine_input = concat_combine_input
        self.cross_attention = cross_attention
        self.implementation = implementation

        # 解码器需要的嵌入层（注意：不是编码器的 enc_linear_embed）
        self.linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)
        self.cond_embed = nn.Embedding(2, hidden_dim)

        # 位置编码
        if self.add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=90
            )
        self.time_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=200)

        # 主干 Transformer
        # 从 TransformerModule.__init__ 中复制 self.transformer 的构造代码
        # （注意：是 self.transformer，不是 self.enc_transformer）

        # 输出头
        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_atom_type_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, atom_dim),
        )

        # cross attention（可选）
        if self.cross_attention:
            # 从 TransformerModule.__init__ 中复制 cross_attention 相关代码

        # concat_combine_input（可选）
        if self.concat_combine_input:
            self.combine_input = nn.Linear(4 * hidden_dim, hidden_dim)

    def decode_z(self, encode_embed, coord_t, atomics_t, padding_mask, t):
        """
        直接从 TransformerModule.decode_z 复制，不做任何修改。
        注意：encode_embed 就是 quant_conv_out(z)，已经投影到 hidden_dim。
        """
        # 完整复制 TransformerModule.decode_z 的方法体
        pass
```

> **重要**：`decode_z` 的方法体请**完整复制**自 `TransformerModule.decode_z`，不要手写，避免引入错误。

---

## 第三步：新建 `CombinedNet`

新建文件 `ENCODER_ROOT/flow_matching/models/combined_net.py`：

```python
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

# 编码器（已在仓库中）
from models.encoder import RelaTransEncoder  # 根据实际路径调整

# 解码器（刚迁移过来的）
from flow_matching.models.components.dae_decoder import DAEDecoder


class CombinedNet(nn.Module):
    """
    桥接 RelaTransEncoder 和 DAEDecoder。
    对外暴露 FlowMatchingModel 需要的三个接口：
        forward(coord_ori, atomics_ori, padding_mask, coord_t, atomics_t, t)
        encode_z(coord_ori, atomics_ori, padding_mask, return_kl=False)
        decode_z(z, coord_t, atomics_t, padding_mask, t)
    """

    def __init__(
        self,
        encoder: RelaTransEncoder,
        decoder: DAEDecoder,
        hidden_dim: int,
        kl_dim: int = 6,
        kl_weight: float = 1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

        # VAE 瓶颈：和原 TransformerModule 中完全一致
        self.quant_conv = nn.Linear(hidden_dim, kl_dim * 2)
        self.quant_conv_out = nn.Linear(kl_dim, hidden_dim)

    # ── 内部工具 ──────────────────────────────────────────────

    def _reparameterize(self, mu, logvar):
        """训练时加噪，推理时取均值。"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def _encode_to_dense(self):
        """
        调用 GNN encoder，输出 padding 格式的节点表示。
        返回：
            h_dense:      [B, N_max, hidden_dim]
            padding_mask: [B, N_max]  True=padding（和 DAE 约定一致）
        """
        data = self._current_data
        # 注意：edge_attr 传 None（已去掉 2D 信息）
        # 如果 RelaTransEncoder.forward 不接受 None，需要传一个全零张量
        h, vec = self.encoder(
            data=data,
            node_feature=data.x,
            edge_index=data.edge_index,
            edge_feature=data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None,
            position=data.pos,
            pos_mask=None,
            return_cls=False,
        )
        h_dense, mask = to_dense_batch(h, data.batch)  # [B, N, H], [B, N]
        padding_mask = ~mask                            # True=padding
        return h_dense, padding_mask

    # ── FlowMatchingModel 需要的三个接口 ─────────────────────

    def encode_z(self, coord_ori, atomics_ori, padding_mask, return_kl=False):
        """推理时由 FlowMatchingModel.encode() 调用。"""
        h_dense, padding_mask_enc = self._encode_to_dense()

        moments = self.quant_conv(h_dense)                   # [B, N, kl_dim*2]
        mu, logvar = torch.chunk(moments, 2, dim=-1)         # 各 [B, N, kl_dim]
        z = self._reparameterize(mu, logvar)                 # [B, N, kl_dim]

        # KL loss，padding 位置不参与计算
        real_mask = (~padding_mask_enc).unsqueeze(-1)        # [B, N, 1]
        kl_item = (1 + logvar - mu.pow(2) - logvar.exp()) * real_mask
        kl_loss = -0.5 * torch.mean(kl_item.sum(dim=-1)) * self.kl_weight

        if return_kl:
            return z, kl_loss
        return z

    def decode_z(self, z, coord_t, atomics_t, padding_mask, t):
        """推理时由 FlowMatchingModel.decode_step() 调用。"""
        encode_embed = self.quant_conv_out(z)                # [B, N, hidden_dim]
        coords, atom_logits = self.decoder.decode_z(
            encode_embed, coord_t, atomics_t, padding_mask, t
        )
        return coords, atom_logits

    def forward(self, coord_ori, atomics_ori, padding_mask, coord_t, atomics_t, t):
        """训练时由 FlowMatchingModel._call_net() 调用。"""
        z, kl_loss = self.encode_z(
            coord_ori, atomics_ori, padding_mask, return_kl=True
        )
        coords, atom_logits = self.decode_z(z, coord_t, atomics_t, padding_mask, t)
        return coords, atom_logits, kl_loss
```

---

## 第四步：新建 Lightning Module

新建文件 `ENCODER_ROOT/flow_matching/trainer/flow_matching_trainer.py`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from tensordict import TensorDict
from torch_geometric.utils import to_dense_batch

from flow_matching.models.flow_matching_model import FlowMatchingModel
from flow_matching.models.combined_net import CombinedNet
from flow_matching.models.components.dae_decoder import DAEDecoder
from models.encoder import RelaTransEncoder  # 根据实际路径调整


class FlowMatchingTrainer(L.LightningModule):

    def __init__(self, args, prior_model=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['prior_model'])

        # ── 构建 encoder ──────────────────────────────────────
        encoder = RelaTransEncoder(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_blocks=args.encoder_blocks,
            prior_model=prior_model,
            args=args,
        )

        # 加载预训练权重（如果有）
        if hasattr(args, 'encoder_ckpt') and args.encoder_ckpt:
            ckpt = torch.load(args.encoder_ckpt, map_location='cpu')
            # checkpoint 可能有不同的 key，根据实际情况选择：
            state_dict = ckpt.get('state_dict', ckpt)
            # 过滤出 encoder 相关的权重
            encoder_state = {
                k.replace('model.encoder.', '').replace('encoder.', ''): v
                for k, v in state_dict.items()
                if 'encoder' in k
            }
            missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
            print(f"Encoder loaded. Missing: {missing}, Unexpected: {unexpected}")

        # 是否冻结 encoder（建议先冻结验证解码器，再端到端微调）
        if getattr(args, 'freeze_encoder', False):
            encoder.requires_grad_(False)
            print("Encoder frozen.")

        # ── 构建 decoder ──────────────────────────────────────
        decoder = DAEDecoder(
            spatial_dim=args.spatial_dim,       # 通常是 3
            atom_dim=args.atom_dim,             # 原子类别数
            hidden_dim=args.hidden_dim,
            num_heads=args.n_heads,
            num_layers=args.decoder_blocks,
            add_sinusoid_posenc=True,
            concat_combine_input=False,
            cross_attention=False,
            implementation='pytorch',
        )

        # ── 组合成 CombinedNet ────────────────────────────────
        net = CombinedNet(
            encoder=encoder,
            decoder=decoder,
            hidden_dim=args.hidden_dim,
            kl_dim=getattr(args, 'kl_dim', 6),
            kl_weight=getattr(args, 'kl_weight', 1e-6),
        )

        # ── 构建 FlowMatchingModel ────────────────────────────
        # 需要根据你的数据集配置 Interpolant
        # 请阅读 tabasco/flow/interpolate.py 确认 Interpolant 的构造参数
        from flow_matching.flow.interpolate import Interpolant
        coords_interpolant = Interpolant(
            # 根据 Interpolant 的实际参数填写
            # 通常需要指定 noise_type, prediction_type 等
        )
        atomics_interpolant = Interpolant(
            # 原子类型插值器，通常和坐标插值器参数不同
        )

        self.flow_model = FlowMatchingModel(
            net=net,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            time_distribution=getattr(args, 'time_distribution', 'uniform'),
            time_alpha_factor=getattr(args, 'time_alpha_factor', 2.0),
            interdist_loss=None,  # 如需要可以开启
            num_random_augmentations=getattr(args, 'num_random_augmentations', None),
            sample_schedule=getattr(args, 'sample_schedule', 'linear'),
        )

        # data_stats 需要在 on_fit_start 里注入
        # 包含: spatial_dim, atom_dim, max_num_atoms, num_atoms_histogram
        self._data_stats = {
            'spatial_dim': args.spatial_dim,
            'atom_dim': args.atom_dim,
            'max_num_atoms': args.max_num_atoms,       # 需要在 args 里指定
            'num_atoms_histogram': {},                  # 在 on_fit_start 里填充
        }

    # ── 接口适配 ──────────────────────────────────────────────

    def _set_current_data(self, batch):
        """把 PyG batch 注入 CombinedNet，供 encode_z/forward 使用。"""
        self.flow_model.net._current_data = batch

    def _to_flow_batch(self, batch):
        """
        把 PyG batch 转换为 FlowMatchingModel 期望的 TensorDict。

        注意事项：
        - data.x 的第几列是原子类型 index，需要根据你的数据集确认
        - atom_dim 需要和 DAEDecoder 里的 atom_dim 一致
        """
        # 坐标：打包格式 → padding 格式
        coords_dense, mask = to_dense_batch(batch.pos, batch.batch)  # [B, N, 3]
        padding_mask = ~mask                                           # [B, N]

        # 原子类型：整数 index → one-hot
        # !! 请确认 data.x 的哪一列是原子类型 index !!
        # 如果 data.x 本身就是整数类型的原子序数：
        atom_types = batch.x[:, 0].long()
        # 如果 data.x 已经是 one-hot，则直接 to_dense_batch 即可

        atomics_packed = F.one_hot(
            atom_types, num_classes=self.args.atom_dim
        ).float()                                                      # [total_nodes, atom_dim]
        atomics_dense, _ = to_dense_batch(
            atomics_packed, batch.batch
        )                                                              # [B, N, atom_dim]

        B = coords_dense.shape[0]
        return TensorDict(
            {
                'coords': coords_dense,
                'atomics': atomics_dense,
                'padding_mask': padding_mask,
            },
            batch_size=B,
        ).to(batch.pos.device)

    # ── Lightning 标准接口 ────────────────────────────────────

    def on_fit_start(self):
        """注入 data_stats，供 FlowMatchingModel 在无条件生成时使用。"""
        # 如果你的 DataModule 有 data_stats，在这里注入：
        # if hasattr(self.trainer.datamodule, 'data_stats'):
        #     self.flow_model.set_data_stats(self.trainer.datamodule.data_stats)
        # 否则用 __init__ 里初始化的 _data_stats：
        self.flow_model.set_data_stats(self._data_stats)

    def training_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        # compute_stats=True 会在训练时额外跑一次采样计算 RMSD，比较慢
        # 建议只在 validation 时开启
        loss, stats = self.flow_model(flow_batch, compute_stats=False)
        self.log('train/loss', loss, batch_size=self.args.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, stats = self.flow_model(flow_batch, compute_stats=True)
        self.log('val/loss', loss, batch_size=self.args.batch_size, prog_bar=True)
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                self.log(f'val/{k}', v, batch_size=self.args.batch_size)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(f'val/{k}', v.item(), batch_size=self.args.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, stats = self.flow_model(flow_batch, compute_stats=True)
        self.log('test/loss', loss, batch_size=self.args.batch_size)
        if 'rmsd' in stats:
            self.log('test/rmsd', stats['rmsd'], batch_size=self.args.batch_size)
        return loss

    def configure_optimizers(self):
        # 只优化 requires_grad=True 的参数（冻结 encoder 时自动排除）
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.args.init_lr,
            weight_decay=self.args.weight_decay,
        )

        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.trainer.fit_loop.setup_data()
            max_iters = (
                self.args.max_epochs
                * len(self.trainer.train_dataloader)
                // self.args.accumulate_grad_batches
            )
            # 从编码器仓库已有的 scheduler 复用
            from your_scheduler_module import LinearWarmupCosineLRSchedulerV2
            scheduler = LinearWarmupCosineLRSchedulerV2(
                optimizer,
                max_iters,
                self.args.min_lr,
                self.args.init_lr,
                self.args.warmup_steps,
                self.args.warmup_lr,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
            }

        elif self.args.scheduler in {'none', 'None'}:
            return optimizer

        else:
            raise NotImplementedError(f"Scheduler '{self.args.scheduler}' not implemented")
```

---

## 第五步：验证顺序

按以下顺序逐步验证，每步通过后再进行下一步：

### 5.1 验证格式转换

```python
# 新建一个临时测试脚本，确认 to_dense_batch 的输出形状正确
from torch_geometric.utils import to_dense_batch

# 用一个 batch 测试
coords_dense, mask = to_dense_batch(batch.pos, batch.batch)
padding_mask = ~mask
print(f"coords_dense: {coords_dense.shape}")   # 期望 [B, N_max, 3]
print(f"padding_mask: {padding_mask.shape}")   # 期望 [B, N_max]
print(f"padding 比例: {padding_mask.float().mean():.2%}")  # 合理范围 0%~50%
```

### 5.2 验证编码器输出

```python
net._current_data = batch
h_dense, padding_mask_enc = net._encode_to_dense()
print(f"h_dense: {h_dense.shape}")       # 期望 [B, N_max, hidden_dim]
print(f"padding_mask: {padding_mask_enc.shape}")  # 期望 [B, N_max]
# 验证 padding 位置的值是否为零
print(f"padding 位置均值: {h_dense[padding_mask_enc].abs().mean():.6f}")  # 应接近 0
```

### 5.3 验证 VAE 瓶颈

```python
z, kl_loss = net.encode_z(None, None, None, return_kl=True)
print(f"z: {z.shape}")          # 期望 [B, N_max, kl_dim]
print(f"kl_loss: {kl_loss}")    # 期望量级在 1e-4 以下（因为 kl_weight=1e-6）
```

### 5.4 验证解码器

```python
# 用随机 z 和 x_t 测试解码器能否正常前向
z_rand = torch.randn(B, N_max, kl_dim).to(device)
coords_t = torch.randn(B, N_max, 3).to(device)
atomics_t = torch.rand(B, N_max, atom_dim).to(device)
t = torch.rand(B).to(device)

coords_pred, atom_logits = net.decode_z(z_rand, coords_t, atomics_t, padding_mask, t)
print(f"coords_pred: {coords_pred.shape}")    # 期望 [B, N_max, 3]
print(f"atom_logits: {atom_logits.shape}")    # 期望 [B, N_max, atom_dim]
```

### 5.5 验证完整 forward

```python
trainer_module._set_current_data(batch)
flow_batch = trainer_module._to_flow_batch(batch)
loss, stats = trainer_module.flow_model(flow_batch, compute_stats=False)
print(f"loss: {loss.item()}")   # 应该是有限值，不是 nan 或 inf
loss.backward()                 # 确认梯度能正常回传
print("梯度回传成功")
```

---

## 注意事项

### edge_attr 为 None 的处理

`RelaTransEncoder.forward` 接收 `edge_feature` 参数。如果传入 `None` 会报错，需要检查 encoder 内部是否有用到 `edge_feature`。如果有，传一个全零张量：

```python
edge_feature = data.edge_attr if (hasattr(data, 'edge_attr') and data.edge_attr is not None) \
               else torch.zeros(data.edge_index.shape[1], args.edge_dim, device=data.pos.device)
```

### hidden_dim 必须一致

`CombinedNet` 的 `quant_conv` 输入维度是 `encoder.hidden_dim`，`quant_conv_out` 输出维度也必须等于 `decoder.hidden_dim`。如果两个仓库的 `hidden_dim` 不同，需要在 `CombinedNet` 里加一个投影层：

```python
# 如果 encoder_hidden_dim != decoder_hidden_dim
self.dim_proj = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
# 在 _encode_to_dense 里：h_dense = self.dim_proj(h_dense)
```

### padding_mask 的 True/False 含义

整个代码里统一约定：**`True` 表示 padding 位置，`False` 表示真实原子**。`to_dense_batch` 返回的 `mask` 含义相反（`True`=真实），所以取反后使用。在所有涉及 `padding_mask` 的地方保持这个约定，避免方向搞反。

### `_current_data` 的线程安全

`_current_data` 是通过实例属性传递 PyG batch 的临时方案。在单 GPU 训练时没有问题。如果使用多 GPU（DDP），每个进程有自己的模型副本，也不会有竞争问题。但如果未来改成更复杂的并行方式，需要重新考虑这个设计。