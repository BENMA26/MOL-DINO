"""
LatentFlowModel: Gaussian Flow Matching in the latent space produced by the
GNN encoder (CombinedNet).

The latent space is a dense tensor of shape [B, N, hidden_dim] where N is the
max number of atoms in the batch and padding positions are masked out.

Training objective (x0-prediction variant of OT flow matching):
    z_t = (1 - t) * z_0 + t * z_1
    loss = MSE( denoiser(z_t, t) - z_1 )   [masked over real atoms only]

Sampling: Euler integration from t=0 (noise) to t=1 (latent).
"""

import math
from copy import deepcopy
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor


class LatentFlowModel(nn.Module):
    """
    Flow Matching model that operates in the GNN encoder's latent space.

    The encoder (frozen CombinedNet) produces:
        z [B, N, hidden_dim]  — per-atom latent representations

    This model learns a flow from Gaussian noise → z using a simple ODE:
        dz/dt = v_θ(z_t, t)

    We use x0-prediction: the network predicts z_1 directly, and the
    velocity is derived as:
        v = (z_pred - z_t) / (1 - t)

    Args:
        denoiser:    nn.Module — the denoiser network (LDMDenoiser)
        hidden_dim:  int — latent dimension (must match encoder hidden_dim)
        min_t:       float — minimum time for training stability
        num_sample_steps: int — Euler steps at inference
        center_latents:   bool — subtract per-molecule CoM from latents
    """

    def __init__(
        self,
        denoiser: nn.Module,
        hidden_dim: int = 256,
        min_t: float = 1e-3,
        num_sample_steps: int = 100,
        center_latents: bool = False,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.hidden_dim = hidden_dim
        self.min_t = min_t
        self.num_sample_steps = num_sample_steps
        self.center_latents = center_latents

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sample_t(self, batch_size: int, device: torch.device) -> Tensor:
        """Uniform t ∈ [min_t, 1]."""
        t = torch.rand(batch_size, device=device)
        return t * (1.0 - self.min_t) + self.min_t        # [B]

    def _sample_noise(
        self, z_1: Tensor, padding_mask: Tensor
    ) -> Tensor:
        """
        Sample Gaussian noise z_0 with same shape as z_1.
        Padded positions are zeroed.

        Args:
            z_1:          [B, N, D]
            padding_mask: [B, N] bool, True = padding

        Returns:
            z_0: [B, N, D]
        """
        z_0 = torch.randn_like(z_1)
        z_0 = z_0 * (~padding_mask).unsqueeze(-1).float()
        return z_0

    def _interpolate(
        self, z_0: Tensor, z_1: Tensor, t: Tensor
    ) -> Tensor:
        """
        Linear interpolation:  z_t = (1 - t) * z_0 + t * z_1

        Args:
            z_0, z_1: [B, N, D]
            t:        [B]

        Returns:
            z_t: [B, N, D]
        """
        t_ = t[:, None, None]                              # [B, 1, 1]
        return (1.0 - t_) * z_0 + t_ * z_1

    def _center(self, z: Tensor, padding_mask: Tensor) -> Tensor:
        """Subtract per-molecule mean over real (non-padding) positions."""
        real = (~padding_mask).unsqueeze(-1).float()       # [B, N, 1]
        n_real = real.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (z * real).sum(dim=1, keepdim=True) / n_real
        return z - mean * real

    # ── training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        z_1: Tensor,
        padding_mask: Tensor,
        compute_stats: bool = False,
    ) -> Tuple[Tensor, Dict]:
        """
        Training forward pass.

        Args:
            z_1:          [B, N, D]  clean latents from frozen encoder
            padding_mask: [B, N]     True = padding position
            compute_stats: bool      whether to return extra diagnostics

        Returns:
            loss:       scalar MSE loss
            stats:      dict of diagnostics (empty if compute_stats=False)
        """
        B, N, D = z_1.shape
        device = z_1.device

        if self.center_latents:
            z_1 = self._center(z_1, padding_mask)

        # Sample time and noise
        t = self._sample_t(B, device)                      # [B]
        z_0 = self._sample_noise(z_1, padding_mask)        # [B, N, D]

        # Interpolate
        z_t = self._interpolate(z_0, z_1, t)              # [B, N, D]

        # Predict clean latent
        z_pred = self.denoiser(z_t, t, padding_mask)      # [B, N, D]

        # MSE loss over real atoms only
        real_mask = (~padding_mask).float()                # [B, N]
        n_real = real_mask.sum(dim=-1).clamp(min=1)        # [B]

        err = (z_pred - z_1) ** 2                          # [B, N, D]
        # per-molecule average: sum over N and D, divide by (#real * D)
        per_mol_loss = (
            (err * real_mask.unsqueeze(-1)).sum(dim=(-1, -2))
            / (n_real * D)
        )                                                   # [B]
        loss = per_mol_loss.mean()

        stats = {}
        if compute_stats:
            stats["ldm_loss"] = loss.item()
            stats["t_mean"] = t.mean().item()
            # Per-time-bin losses
            t_cpu = t.detach().cpu().numpy()
            loss_cpu = per_mol_loss.detach().cpu().numpy()
            bins = np.linspace(0, 1, 5)
            for i in range(len(bins) - 1):
                mask_b = (t_cpu >= bins[i]) & (t_cpu < bins[i + 1])
                if mask_b.sum() > 0:
                    stats[f"ldm_loss_t[{bins[i]:.1f},{bins[i+1]:.1f})"] = (
                        loss_cpu[mask_b].mean()
                    )

        return loss, stats

    # ── sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        padding_mask: Tensor,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        """
        Generate latents from Gaussian noise via Euler integration.

        Args:
            padding_mask:  [B, N] bool  — True = padding
            num_steps:     int          — number of Euler steps (default: self.num_sample_steps)
            return_trajectory: bool     — if True, also return list of z_t at each step

        Returns:
            z_1_pred: [B, N, D]  — generated latents at t=1
            (optionally) trajectory list
        """
        if num_steps is None:
            num_steps = self.num_sample_steps

        B, N = padding_mask.shape
        device = padding_mask.device

        # Start from Gaussian noise
        z_t = torch.randn(B, N, self.hidden_dim, device=device)
        z_t = z_t * (~padding_mask).unsqueeze(-1).float()

        ts = torch.linspace(self.min_t, 1.0, num_steps + 1, device=device)
        trajectory = [z_t.clone()] if return_trajectory else []

        for i in range(num_steps):
            t_curr = ts[i].expand(B)                       # [B]
            dt = ts[i + 1] - ts[i]                        # scalar

            z_pred = self.denoiser(z_t, t_curr, padding_mask)  # [B, N, D]

            # Euler step: v = (z_pred - z_t) / (1 - t)
            denom = (1.0 - t_curr[:, None, None]).clamp(min=1e-6)
            velocity = (z_pred - z_t) / denom
            z_t = z_t + velocity * dt

            # Zero padding
            z_t = z_t * (~padding_mask).unsqueeze(-1).float()

            if return_trajectory:
                trajectory.append(z_t.clone())

        if return_trajectory:
            return z_t, trajectory
        return z_t
