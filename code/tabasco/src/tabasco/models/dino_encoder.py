"""
DinoEncoderModule: wraps the 3D-GSRD RelaTransEncoder so it can be used
as a drop-in replacement for the Tabasco TransformerModule encoder.

Interface contract (same as TransformerModule):
  encode_z(coords, atomics, padding_mask) -> z  [B, N, kl_dim]
  decode_z(z, coord_t, atomics_t, padding_mask, t) -> (coords, atom_logits)
  forward(coord_ori, atomics_ori, padding_mask, coord_t, atomics_t, t) -> (coords, atom_logits, kl_loss)

The 3D-GSRD encoder expects PyG sparse-graph inputs, so we convert
dense (B, N, F) tensors to a batched PyG Data object on the fly.
"""

import sys
import os
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean

# ── positional / time encodings reused from Tabasco ──────────────────────────
from tabasco.models.components.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)


# ---------------------------------------------------------------------------
# Helper: dense (B,N,F) → PyG Batch
# ---------------------------------------------------------------------------

def dense_to_pyg_batch(coords: Tensor, atomics: Tensor, padding_mask: Tensor) -> Batch:
    """Convert dense padded tensors to a PyG Batch with full-graph edges.

    Args:
        coords:       (B, N, 3)  – 3-D coordinates
        atomics:      (B, N, A)  – one-hot atom types
        padding_mask: (B, N)     – True where padded (no atom)

    Returns:
        PyG Batch with fields: x, pos, edge_index, edge_attr, batch, z
    """
    B, N, _ = coords.shape
    data_list = []
    for b in range(B):
        mask = ~padding_mask[b]          # True for real atoms
        pos_b = coords[b][mask]          # (n, 3)
        atom_b = atomics[b][mask]        # (n, A)
        n = pos_b.shape[0]

        # full-graph edges (all pairs, no self-loops)
        src, dst = torch.meshgrid(
            torch.arange(n, device=coords.device),
            torch.arange(n, device=coords.device),
            indexing="ij",
        )
        mask_edge = src != dst
        edge_index = torch.stack([src[mask_edge], dst[mask_edge]], dim=0)  # (2, E)

        # edge_attr: one-hot bond type placeholder (all zeros = "no bond info")
        # 3D-GSRD edge_embedding expects (edge_dim,) features; we use zeros
        # so the model relies purely on distance features.
        edge_attr = torch.zeros(edge_index.shape[1], 1, device=coords.device)

        # atomic number z (argmax of one-hot, 1-indexed to avoid 0=padding)
        z = atom_b.argmax(dim=-1) + 1   # (n,)

        data = Data(
            x=atom_b.float(),
            pos=pos_b.float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            z=z,
        )
        data_list.append(data)

    return Batch.from_data_list(data_list)


# ---------------------------------------------------------------------------
# Projection head: node-level hidden_dim → per-node kl_dim (VAE bottleneck)
# ---------------------------------------------------------------------------

class NodeVAEBottleneck(nn.Module):
    """Per-node VAE reparameterisation on top of the DINO encoder output."""

    def __init__(self, hidden_dim: int, kl_dim: int, kl_weight: float = 1e-6):
        super().__init__()
        self.kl_weight = kl_weight
        self.quant_conv = nn.Linear(hidden_dim, kl_dim * 2)
        self.quant_conv_out = nn.Linear(kl_dim, hidden_dim)

    def encode(self, h: Tensor, padding_mask: Tensor, training: bool) -> Tuple[Tensor, Tensor]:
        """h: (B, N, hidden_dim) dense; returns z (B, N, kl_dim), kl_loss scalar."""
        moments = self.quant_conv(h)
        mu, logvar = torch.chunk(moments, 2, dim=-1)
        if training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        real_mask = ~padding_mask                          # (B, N)
        kl_item = (1 + logvar - mu.pow(2) - logvar.exp()) * real_mask.unsqueeze(-1)
        kl_loss = -0.5 * torch.mean(kl_item.sum(dim=-1)) * self.kl_weight
        return z, kl_loss

    def project_out(self, z: Tensor) -> Tensor:
        return self.quant_conv_out(z)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DinoEncoderModule(nn.Module):
    """Tabasco-compatible module that uses the 3D-GSRD RelaTransEncoder.

    The encoder is loaded from a DINO pre-training checkpoint.
    The decoder is the same lightweight Transformer decoder as in
    TransformerModule (decode_z path).
    """

    def __init__(
        self,
        # ── 3D-GSRD encoder args ──────────────────────────────────────────
        gsrd_args,                        # argparse Namespace from 3D-GSRD
        gsrd_checkpoint_path: str,        # path to DINO pretrain .ckpt
        # ── bottleneck ───────────────────────────────────────────────────
        kl_dim: int = 6,
        kl_weight: float = 1e-6,
        # ── decoder (same as TransformerModule) ──────────────────────────
        spatial_dim: int = 3,
        atom_dim: int = 16,
        num_heads: int = 8,
        num_layers: int = 8,
        hidden_dim: int = 256,
        add_sinusoid_posenc: bool = True,
        train_diffusion: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.train_diffusion = train_diffusion
        self.freeze_encoder = freeze_encoder

        # ── build 3D-GSRD encoder ─────────────────────────────────────────
        # Import here to avoid polluting the global namespace
        _gsrd_root = os.path.join(
            os.path.dirname(__file__),
            "../../../../../../3D-GSRD",
        )
        _gsrd_root = os.path.normpath(_gsrd_root)
        if _gsrd_root not in sys.path:
            sys.path.insert(0, _gsrd_root)

        from model.retrans import RelaTransEncoder  # noqa: PLC0415
        from model.autoencoder import AutoEncoder   # noqa: PLC0415
        from training_utils import load_encoder_params  # noqa: PLC0415

        self.gsrd_encoder = RelaTransEncoder(
            node_dim=gsrd_args.node_dim,
            edge_dim=gsrd_args.edge_dim,
            hidden_dim=gsrd_args.hidden_dim,
            n_heads=gsrd_args.n_heads,
            n_blocks=gsrd_args.encoder_blocks,
            prior_model=None,
            args=gsrd_args,
        )
        # load pretrained weights
        load_encoder_params(self.gsrd_encoder, gsrd_checkpoint_path)

        if freeze_encoder:
            for p in self.gsrd_encoder.parameters():
                p.requires_grad_(False)

        gsrd_hidden = gsrd_args.hidden_dim

        # projection: gsrd_hidden → tabasco hidden_dim (if different)
        self.enc_proj = (
            nn.Linear(gsrd_hidden, hidden_dim, bias=False)
            if gsrd_hidden != hidden_dim
            else nn.Identity()
        )

        # ── VAE bottleneck ────────────────────────────────────────────────
        self.bottleneck = NodeVAEBottleneck(hidden_dim, kl_dim, kl_weight)

        # ── decoder (mirrors TransformerModule.decode_z) ──────────────────
        self.cond_embed = nn.Embedding(2, hidden_dim)
        self.linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)

        if add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=90
            )
        else:
            self.positional_encoding = None

        self.time_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=200)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation=nn.SiLU(inplace=False),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=num_layers)

        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_atom_type_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, atom_dim),
        )

    # ------------------------------------------------------------------
    # encode_z: dense → DINO encoder → VAE → z  (B, N, kl_dim)
    # ------------------------------------------------------------------

    def encode_z(
        self,
        coord_ori: Tensor,
        atomics_ori: Tensor,
        padding_mask: Tensor,
        return_kl: bool = False,
    ) -> Tensor:
        B, N, _ = coord_ori.shape

        # 1. convert to PyG batch
        pyg_batch = dense_to_pyg_batch(coord_ori, atomics_ori, padding_mask)
        pyg_batch = pyg_batch.to(coord_ori.device)

        # 2. run GSRD encoder  →  (total_real_nodes, gsrd_hidden), vec ignored
        rep, _vec = self.gsrd_encoder(
            data=pyg_batch,
            node_feature=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            edge_feature=pyg_batch.edge_attr,
            position=pyg_batch.pos,
            pos_mask=None,
        )

        # 3. scatter back to dense (B, N, gsrd_hidden)
        rep_dense, _ = to_dense_batch(rep, pyg_batch.batch, max_num_nodes=N)

        # 4. project to tabasco hidden_dim
        rep_dense = self.enc_proj(rep_dense)

        # 5. VAE bottleneck
        z, kl_loss = self.bottleneck.encode(rep_dense, padding_mask, self.training)

        if return_kl:
            return z, kl_loss
        return z

    # ------------------------------------------------------------------
    # decode_z: z + noisy (coord_t, atomics_t) → velocity prediction
    # ------------------------------------------------------------------

    def decode_z(
        self,
        z: Tensor,
        coord_t: Tensor,
        atomics_t: Tensor,
        padding_mask: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        real_mask = (~padding_mask).float()          # (B, N)

        encode_embed = self.bottleneck.project_out(z)  # (B, N, hidden_dim)

        embed_coords = self.linear_embed(coord_t)
        embed_atom_types = self.atom_type_embed(atomics_t.argmax(dim=-1))

        if self.positional_encoding is not None:
            embed_posenc = self.positional_encoding(
                batch_size=coord_t.shape[0], seq_len=coord_t.shape[1]
            )
        else:
            embed_posenc = torch.zeros_like(embed_coords)

        encode_embed = (
            encode_embed * real_mask.unsqueeze(-1)
            + embed_posenc * real_mask.unsqueeze(-1)
        )

        embed_time = self.time_encoding(t).unsqueeze(1)  # (B, 1, hidden_dim)

        h_in = embed_coords + embed_atom_types + embed_posenc + embed_time
        h_in = h_in * real_mask.unsqueeze(-1)

        # concat latent context (same trick as TransformerModule)
        h_in = torch.cat([h_in, encode_embed], dim=1)   # (B, 2N, hidden_dim)

        B, N_orig = coord_t.shape[:2]
        cond_pos = torch.cat(
            [
                torch.zeros(B, N_orig, dtype=torch.long, device=coord_t.device),
                torch.ones(B, N_orig, dtype=torch.long, device=coord_t.device),
            ],
            dim=1,
        )
        h_in = h_in + self.cond_embed(cond_pos)

        double_mask = torch.cat([padding_mask, padding_mask], dim=1)
        h_out = self.transformer(h_in, src_key_padding_mask=double_mask)

        h_out = h_out[:, :N_orig, :] * real_mask.unsqueeze(-1)

        coords = self.out_coord_linear(h_out)
        atom_logits = self.out_atom_type_linear(h_out)
        return coords, atom_logits

    # ------------------------------------------------------------------
    # forward: used during autoencoder training
    # ------------------------------------------------------------------

    def forward(
        self,
        coord_ori: Tensor,
        atomics_ori: Tensor,
        padding_mask: Tensor,
        coord_t: Tensor,
        atomics_t: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        z, kl_loss = self.encode_z(coord_ori, atomics_ori, padding_mask, return_kl=True)
        coords, atom_logits = self.decode_z(z, coord_t, atomics_t, padding_mask, t)
        return coords, atom_logits, kl_loss
