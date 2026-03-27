"""
DAEDecoder: 从 TransformerModule 中提取的解码器部分。
只保留 decode_z 需要的组件，去掉编码器相关的所有内容。
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


# ── 位置编码 ──────────────────────────────────────────────────────────────────

class SinusoidEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, posenc_dim, max_len=100):
        super().__init__()
        self.posenc_dim = posenc_dim
        self.max_len = max_len

        pos_embed = torch.zeros(max_len, posenc_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, posenc_dim, 2).float()
            * (-math.log(10 * max_len) / posenc_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, batch_size: int, seq_len: int) -> Tensor:
        pos_embed = self.pos_embed[:, :seq_len, :]
        return pos_embed.expand(batch_size, -1, -1)


class TimeFourierEncoding(nn.Module):
    """Encoder for continuous timesteps in [0, 1]."""

    def __init__(self, posenc_dim, max_len=100):
        super().__init__()
        self.posenc_dim = posenc_dim
        self.max_len = max_len

    def forward(self, t: Tensor) -> Tensor:
        import torch.nn.functional as F
        t_scaled = t * self.max_len
        half_dim = self.posenc_dim // 2
        emb = math.log(self.max_len) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb
        )
        emb = torch.outer(t_scaled.float(), emb)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.posenc_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return emb


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        import torch.nn.functional as F
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


# ── DAEDecoder ────────────────────────────────────────────────────────────────

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
        num_layers: int,
        activation: str = "SiLU",
        add_sinusoid_posenc: bool = True,
        concat_combine_input: bool = False,
        cross_attention: bool = False,
        implementation: str = "pytorch",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.add_sinusoid_posenc = add_sinusoid_posenc
        self.concat_combine_input = concat_combine_input
        self.cross_attention = cross_attention
        self.implementation = implementation

        # 解码器嵌入层
        self.linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)
        self.cond_embed = nn.Embedding(2, hidden_dim)

        # 位置编码
        if self.add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=90
            )
        self.time_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=200)

        # 激活函数
        if activation == "SiLU":
            act = nn.SiLU(inplace=False)
        elif activation == "ReLU":
            act = nn.ReLU(inplace=False)
        elif activation == "SwiGLU":
            act = SwiGLU()
        else:
            raise ValueError(f"Invalid activation: {activation}")

        # 主干 Transformer
        if self.implementation == "pytorch":
            diff_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                activation=act,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(diff_layer, num_layers=num_layers)
        elif self.implementation == "reimplemented":
            from flow_matching.models.components.transformer import Transformer
            self.transformer = Transformer(
                dim=hidden_dim,
                num_heads=num_heads,
                depth=num_layers,
            )
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")

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

        # concat_combine_input（可选）
        if self.concat_combine_input:
            self.combine_input = nn.Linear(4 * hidden_dim, hidden_dim)

    def decode_z(self, encode_embed, coord_t, atomics_t, padding_mask, t) -> tuple:
        """
        直接从 TransformerModule.decode_z 复制，不做任何修改。
        encode_embed 已经是 quant_conv_out(z)，投影到 hidden_dim。
        """
        real_mask = 1 - padding_mask.int()

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
        encode_embed = encode_embed * real_mask.unsqueeze(-1) + embed_posenc * real_mask.unsqueeze(-1)

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
        h_in = torch.cat([h_in, encode_embed], dim=1)
        cond_pos = torch.cat(
            (
                torch.zeros(coord_t.shape[0], coord_t.shape[1], dtype=torch.long, device=coord_t.device),
                torch.ones(coord_t.shape[0], coord_t.shape[1], dtype=torch.long, device=coord_t.device),
            ),
            dim=-1,
        )
        h_in = h_in + self.cond_embed(cond_pos)

        if self.implementation == "pytorch":
            h_out = self.transformer(h_in, src_key_padding_mask=torch.cat([padding_mask, padding_mask], dim=1))
        elif self.implementation == "reimplemented":
            h_out = self.transformer(h_in, padding_mask=torch.cat([padding_mask, padding_mask], dim=1))

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
