"""
CombinedNet: 桥接 RelaTransEncoder 和 DAEDecoder。
"""
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from model.retrans import RelaTransEncoder
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

        # VAE 瓶颈
        self.quant_conv = nn.Linear(hidden_dim, kl_dim * 2)
        self.quant_conv_out = nn.Linear(kl_dim, hidden_dim)

        # 用于在 training_step 中注入当前 batch
        self._current_data = None

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
            padding_mask: [B, N_max]  True=padding
        """
        data = self._current_data

        # edge_attr: 如果不存在或为 None，传全零张量
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_feature = data.edge_attr
        else:
            edge_dim = self.encoder.edge_embedding[0].in_features - 1  # edge_dim+1 -> edge_dim
            edge_feature = torch.zeros(
                data.edge_index.shape[1], edge_dim,
                device=data.pos.device, dtype=data.pos.dtype
            )

        h, vec = self.encoder(
            data=data,
            node_feature=data.x,
            edge_index=data.edge_index,
            edge_feature=edge_feature,
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
