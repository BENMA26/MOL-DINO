"""
LDM Denoiser: Transformer-based denoiser operating in the latent space.

Input:  noisy latent z_t of shape [B, N, hidden_dim] + timestep t [B]
Output: predicted clean latent z_1 of shape [B, N, hidden_dim]

Architecture: standard pre-norm Transformer with sinusoidal time embedding
injected via additive conditioning (AdaLN-zero style simplified to additive shift).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ── Positional / time encodings ───────────────────────────────────────────────

class SinusoidalPosEnc(nn.Module):
    """Classic sinusoidal positional encoding for sequence positions."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()         # [L, 1]
        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )                                                         # [dim/2]
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))              # [1, L, dim]

    def forward(self, seq_len: int) -> Tensor:
        return self.pe[:, :seq_len, :]                           # [1, N, dim]


class TimestepEmbedder(nn.Module):
    """Embed scalar timestep t ∈ [0,1] into a vector of shape [B, dim]."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        # MLP to project Fourier features → dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: [B] float in [0, 1]
        Returns:
            [B, dim] time embedding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device).float()
            / half
        )                                                        # [half]
        t_scaled = t[:, None] * freqs[None, :]                  # [B, half]
        emb = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)                                     # [B, dim]


# ── Main denoiser ─────────────────────────────────────────────────────────────

class LDMDenoiser(nn.Module):
    """
    Transformer denoiser for the latent space of shape [B, N, hidden_dim].

    Takes:
        z_t         [B, N, hidden_dim]  — noisy latent at time t
        t           [B]                 — noise time in [0, 1]
        padding_mask [B, N]             — True = padding (ignored position)

    Returns:
        z_pred      [B, N, hidden_dim]  — predicted clean z_1 (x0-prediction)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embedder = TimestepEmbedder(hidden_dim)

        # Sequence positional encoding
        self.pos_enc = SinusoidalPosEnc(hidden_dim, max_len=max_seq_len)

        # Input projection (latent may already be hidden_dim, but add a linear
        # layer for flexibility and to allow easy dimension changes)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Time conditioning: project time emb → scale/shift per token
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),   # scale + shift
        )

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize output projection near zero for training stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z_t:          [B, N, hidden_dim]  noisy latent
            t:            [B]                  time in [0, 1]
            padding_mask: [B, N] bool, True = padding

        Returns:
            z_pred: [B, N, hidden_dim]  predicted clean z_1
        """
        B, N, D = z_t.shape

        # 1. Input projection + positional encoding
        h = self.input_proj(z_t)                             # [B, N, D]
        h = h + self.pos_enc(N).to(h.device)                # [B, N, D]

        # 2. Time conditioning: additive shift + scale
        t_emb = self.time_embedder(t)                        # [B, D]
        t_cond = self.time_proj(t_emb)                       # [B, 2*D]
        scale, shift = t_cond.chunk(2, dim=-1)               # [B, D] each
        scale = scale.unsqueeze(1)                           # [B, 1, D]
        shift = shift.unsqueeze(1)                           # [B, 1, D]
        h = h * (1 + scale) + shift                          # [B, N, D]

        # 3. Transformer (mask out padding positions)
        h = self.transformer(h, src_key_padding_mask=padding_mask)  # [B, N, D]

        # 4. Output
        h = self.output_norm(h)
        z_pred = self.output_proj(h)                         # [B, N, D]

        # Zero out padded positions
        if padding_mask is not None:
            z_pred = z_pred * (~padding_mask).unsqueeze(-1).float()

        return z_pred
