import sys
import os
import argparse
from typing import Optional

import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_dense_batch

# 把 3D-GSRD 加入 path 以便 import
_GSRD_DIR = os.path.join(os.path.dirname(__file__), "../../../../../3D-GSRD")
if _GSRD_DIR not in sys.path:
    sys.path.insert(0, _GSRD_DIR)

from model.retrans import RelaTransEncoder  # noqa: E402

from tabasco.models.components.common import SwiGLU
from tabasco.models.components.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from tabasco.models.components.transformer import Transformer

class TransformerModule(nn.Module):
    """Basic Transformer model for molecule generation."""

    def __init__(
        self,
        spatial_dim: int,
        atom_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        activation: str = "SiLU",
        implementation: str = "pytorch",  # "pytorch" "reimplemented"
        cross_attention: bool = False,
        add_sinusoid_posenc: bool = True,
        concat_combine_input: bool = False,
        custom_weight_init: Optional[str] = None,
        # 3D-GSRD encoder 参数
        gsrd_node_dim: int = 63,
        gsrd_edge_dim: int = 4,
        gsrd_hidden_dim: int = 256,
        gsrd_n_heads: int = 8,
        gsrd_encoder_blocks: int = 8,
        gsrd_radius: float = 2.5,
        gsrd_checkpoint: Optional[str] = None,
    ):
        """
        Args:
            custom_weight_init: None, "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal"
            (uniform does not work well)
            gsrd_*: 3D-GSRD RelaTransEncoder 的超参数
            gsrd_radius: 动态建图的距离截断半径（Angstrom，注意 TABASCO 坐标已除以 2.0）
            gsrd_checkpoint: 预训练权重路径，None 则随机初始化
        """
        super().__init__()

        # Normalize custom_weight_init if it's the string "None"
        if isinstance(custom_weight_init, str) and custom_weight_init.lower() == "none":
            custom_weight_init = None

        self.input_dim = spatial_dim + atom_dim
        self.time_dim = 1
        self.comb_input_dim = self.input_dim + self.time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.implementation = implementation
        self.cross_attention = cross_attention
        self.add_sinusoid_posenc = add_sinusoid_posenc
        self.concat_combine_input = concat_combine_input
        self.custom_weight_init = custom_weight_init
        self.gsrd_radius = gsrd_radius
        print(f"Implementation: {self.implementation}")

        self.cond_embed = nn.Embedding(2, hidden_dim)

        # ── 3D-GSRD encoder（固定，不参与梯度更新）────────────────────────────
        gsrd_args = argparse.Namespace(
            hidden_dim=gsrd_hidden_dim,
            dropout=0.0,
            pair_update=False,
            trans_version="v6",
            attn_activation="silu",
            dataset="pcqm4mv2",
            dataset_arg=None,
            use_cls_token=False,
            cls_distance=1.0,
        )
        self.gsrd_encoder = RelaTransEncoder(
            node_dim=gsrd_node_dim,
            edge_dim=gsrd_edge_dim,
            hidden_dim=gsrd_hidden_dim,
            n_heads=gsrd_n_heads,
            n_blocks=gsrd_encoder_blocks,
            prior_model=None,
            args=gsrd_args,
        )
        if gsrd_checkpoint is not None:
            state = torch.load(gsrd_checkpoint, map_location="cpu", weights_only=False)
            # 兼容 lightning checkpoint 和裸 state_dict
            state = state.get("state_dict", state)
            # 只加载 encoder 部分的权重
            enc_state = {
                k.replace("model.encoder.", ""): v
                for k, v in state.items()
                if k.startswith("model.encoder.")
            }
            self.gsrd_encoder.load_state_dict(enc_state, strict=False)
            print(f"Loaded GSRD encoder weights from {gsrd_checkpoint}")
        # 冻结
        for p in self.gsrd_encoder.parameters():
            p.requires_grad = False
        self.gsrd_encoder.eval()

        # atom_dim → gsrd_node_dim 的节点特征投影（可训练）
        # GSRD node_embedding 接收 node_dim+3 维输入（atomics concat pos），
        # 这里先把 tabasco 的 atom_dim 投影到 gsrd_node_dim 以对齐
        self.node_feat_proj = nn.Linear(atom_dim, gsrd_node_dim)

        # gsrd_hidden_dim → hidden_dim 的输出投影（可训练）
        self.enc_out_proj = (
            nn.Linear(gsrd_hidden_dim, hidden_dim)
            if gsrd_hidden_dim != hidden_dim
            else nn.Identity()
        )
        # ─────────────────────────────────────────────────────────────────────

        self.linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)

        if self.add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=90
            )

        if self.concat_combine_input:
            self.combine_input = nn.Linear(4 * hidden_dim, hidden_dim)

        self.time_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=200)

        if activation == "SiLU":
            activation = nn.SiLU(inplace=False)
        elif activation == "ReLU":
            activation = nn.ReLU(inplace=False)
        elif activation == "SwiGLU":
            activation = SwiGLU()
        else:
            raise ValueError(f"Invalid activation: {activation}")

        if self.implementation == "pytorch":
            diff_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                diff_layer, num_layers=num_layers
            )
        elif self.implementation == "reimplemented":
            self.transformer = Transformer(
                dim=hidden_dim,
                num_heads=num_heads,
                depth=num_layers,
            )
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")
        # self.quant_conv_out_norm = nn.LayerNorm(6)
        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )

        self.out_atom_type_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, atom_dim),
        )

        # Add cross attention layers
        if self.cross_attention:
            self.coord_cross_attention = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )

            self.atom_cross_attention = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )

        self._atom_size_tuples = []

        if self.custom_weight_init is not None:
            print(f"Initializing weights with {self.custom_weight_init}!!!!")
            self.apply(self._custom_weight_init)

    def _custom_weight_init(self, module):
        """Initialize the weights of the module with a custom method."""
        for name, param in module.named_parameters():
            if "weight" in name and param.data.dim() >= 2:
                if self.custom_weight_init == "xavier":
                    nn.init.xavier_uniform_(param)
                elif self.custom_weight_init == "kaiming":
                    nn.init.kaiming_uniform_(param)
                elif self.custom_weight_init == "orthogonal":
                    nn.init.orthogonal_(param)
                elif self.custom_weight_init == "uniform":
                    nn.init.uniform_(param)
                elif self.custom_weight_init == "eye":
                    nn.init.eye_(param)
                elif self.custom_weight_init == "normal":
                    nn.init.normal_(param)
                else:
                    raise ValueError(
                        f"Invalid custom weight init: {self.custom_weight_init}"
                    )

    def encode(self, coord_ori, atomics_ori, padding_mask):
        """用固定的 3D-GSRD RelaTransEncoder 编码原始分子。

        流程：
          1. dense (B, N, D) → sparse (N_total, D)，去掉 padding
          2. 节点特征投影 atom_dim → gsrd_node_dim
          3. 根据 3D 坐标动态构建 radius graph
          4. 调用冻结的 RelaTransEncoder
          5. sparse → dense (B, N_max, gsrd_hidden_dim)
          6. 输出维度投影 gsrd_hidden_dim → hidden_dim
        """
        real_mask = ~padding_mask                           # (B, N)  True = 真实原子

        # ── 1. dense → sparse ────────────────────────────────────────────────
        batch_idx = real_mask.nonzero(as_tuple=False)[:, 0]  # (N_total,)
        pos = coord_ori[real_mask]                            # (N_total, 3)
        atom_feats = atomics_ori[real_mask].float()           # (N_total, atom_dim)

        # ── 2. 节点特征投影 ───────────────────────────────────────────────────
        node_feat = self.node_feat_proj(atom_feats)           # (N_total, gsrd_node_dim)

        # ── 3. 动态建图（radius graph）────────────────────────────────────────
        edge_index = radius_graph(
            pos, r=self.gsrd_radius, batch=batch_idx, loop=False
        )                                                     # (2, E)

        # edge_attr: edge_embedding 接收 edge_dim+1 维（bond特征+距离），
        # 我们没有键型信息，直接用全零+L2距离拼成匹配的维度
        in_features = self.gsrd_encoder.edge_embedding[0].in_features  # = edge_dim + 1
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)         # (E, 1)
        edge_attr = torch.cat(
            [pos.new_zeros(edge_index.shape[1], in_features - 1), dist], dim=-1
        )                                                                # (E, edge_dim+1)

        # ── 4. 构造 data 对象，调用冻结 encoder ──────────────────────────────
        data = argparse.Namespace(batch=batch_idx, ptr=None)
        with torch.no_grad():
            h, _ = self.gsrd_encoder(
                data=data,
                node_feature=node_feat,
                edge_index=edge_index,
                edge_feature=edge_attr,
                position=pos,
            )                                                 # h: (N_total, gsrd_hidden_dim)

        # ── 5. sparse → dense ────────────────────────────────────────────────
        max_num_atoms = coord_ori.shape[1]
        h_dense, _ = to_dense_batch(
            h, batch_idx, max_num_nodes=max_num_atoms
        )                                                     # (B, N_max, gsrd_hidden_dim)

        # ── 6. 维度投影 ───────────────────────────────────────────────────────
        return self.enc_out_proj(h_dense)                     # (B, N_max, hidden_dim)

    def forward(self, coord_ori, atomics_ori, padding_mask, coord_t, atomics_t, t, mode = "val") -> Tensor:
        """Forward pass of the module."""

        real_mask = 1 - padding_mask.int()
        encode_embed = self.encode(coord_ori, atomics_ori, padding_mask) * real_mask.unsqueeze(-1)

        embed_coords = self.linear_embed(coord_t)
        embed_atom_types = self.atom_type_embed(atomics_t.argmax(dim=-1))

        if self.add_sinusoid_posenc:
            embed_posenc = self.positional_encoding(
                batch_size=coord_t.shape[0], seq_len=coord_t.shape[1]
            )
        else:
            embed_posenc = torch.zeros(
                coord_t.shape[0], coord_t.shape[1], self.hidden_dim
            ).to(coord_t.device)

        embed_time = self.time_encoding(t).unsqueeze(1)

        assert embed_posenc.shape == embed_coords.shape == embed_atom_types.shape, (
            f"embed_posenc.shape: {embed_posenc.shape}, embed_coords.shape: {embed_coords.shape}, embed_atom_types.shape: {embed_atom_types.shape}"
        )

        if self.concat_combine_input:
            embed_time = embed_time.repeat(1, coord_t.shape[1], 1)
            h_in = torch.cat(
                [embed_coords, embed_atom_types, embed_posenc, embed_time], dim=-1
            )
            assert h_in.shape == (
                coord_t.shape[0],
                coord_t.shape[1],
                4 * self.hidden_dim,
            ), f"h_in.shape: {h_in.shape}"
            h_in = self.combine_input(h_in)
            assert h_in.shape == (coord_t.shape[0], coord_t.shape[1], self.hidden_dim), (
                f"h_in.shape: {h_in.shape}"
            )
        else:
            h_in = embed_coords + embed_atom_types + embed_posenc + embed_time
        h_in = h_in * real_mask.unsqueeze(-1)
        # h_in = h_in + encode_embed
        h_in = torch.cat( [h_in, encode_embed], dim=1)
        cond_pos = torch.cat(
                (
                    torch.zeros(coord_t.shape[0], coord_t.shape[1], dtype=torch.long, device=coord_t.device),
                    torch.ones(coord_t.shape[0], coord_t.shape[1], dtype=torch.long, device=coord_t.device),
                ),
                dim=-1,
            )
        h_in = h_in + self.cond_embed(cond_pos)
        
        if self.implementation == "pytorch":
            h_out = self.transformer(h_in, src_key_padding_mask=torch.cat( [padding_mask, padding_mask], dim=1))
        elif self.implementation == "reimplemented":
            h_out = self.transformer(h_in, padding_mask=torch.cat( [padding_mask, padding_mask], dim=1))

        # load the half of the input to the output
        seq_len = coord_t.shape[1]
        h_out = h_out[:, :seq_len, :]
        h_out = h_out * real_mask.unsqueeze(-1)

        if self.cross_attention:
            h_coord = self.coord_cross_attention(
                h_out,
                h_in[:, :seq_len, :],
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            coords = self.out_coord_linear(h_coord)
        else:
            coords = self.out_coord_linear(h_out)

        if self.cross_attention:
            h_atom = self.atom_cross_attention(
                h_out,
                h_in[:, :seq_len, :],
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            atom_logits = self.out_atom_type_linear(h_atom)
        else:
            atom_logits = self.out_atom_type_linear(h_out)

        return coords, atom_logits
