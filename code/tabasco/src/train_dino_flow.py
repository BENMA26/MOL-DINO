"""
Train flow matching model in the latent space of DINO autoencoder.

Usage:
    python train_dino_flow.py \
        --ae_ckpt ./outputs/dino_ae_stage2/best.ckpt \
        --data_dir /path/to/crossdocked_train.pt \
        --val_data_dir /path/to/crossdocked_val.pt \
        --output_dir ./outputs/dino_flow \
        --max_epochs 200 \
        --batch_size 128
"""

import argparse
import os
import sys

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.optim import AdamW
from tensordict import TensorDict

_here = os.path.dirname(os.path.abspath(__file__))
_tabasco_src = _here
if _tabasco_src not in sys.path:
    sys.path.insert(0, _tabasco_src)

from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from tabasco.models.flow_model import FlowMatchingModel
from tabasco.flow.interpolate import VPInterpolant, OTInterpolant
from train_dino_ae import DinoAELitModule


# ── Lightning module for flow matching in latent space ────────────────────────

class DinoFlowLitModule(L.LightningModule):
    def __init__(
        self,
        ae_ckpt: str,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # load frozen autoencoder
        self.ae = DinoAELitModule.load_from_checkpoint(ae_ckpt, map_location="cpu")
        self.ae.eval()
        self.ae.requires_grad_(False)

        # build flow matching model (operates on latent z)
        from tabasco.models.components.transformer import TransformerModule

        kl_dim = self.ae.net.bottleneck.quant_conv.out_features // 2
        spatial_dim = 3
        atom_dim = 16  # match train_dino_ae.py

        denoiser = TransformerModule(
            hidden_dim=hidden_dim,
            kl_dim=kl_dim,
            spatial_dim=spatial_dim,
            atom_dim=atom_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        coords_interpolant = OTInterpolant(
            data_scale=1.0,
            noise_scale=1.0,
        )
        atomics_interpolant = VPInterpolant(
            beta_min=0.1,
            beta_max=20.0,
        )

        self.flow_model = FlowMatchingModel(
            net=denoiser,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            time_distribution="uniform",
            num_random_augmentations=0,
        )

    def _step(self, batch, stage: str):
        coords = batch["coords"]
        atomics = batch["atomics"]
        padding_mask = batch["padding_mask"]

        # encode to latent with frozen AE
        with torch.no_grad():
            z = self.ae.net.encode_z(coords, atomics, padding_mask, return_kl=False)

        # create latent batch for flow model
        latent_batch = TensorDict(
            {
                "coords": z,  # treat latent as "coords" for flow model
                "atomics": atomics,
                "padding_mask": padding_mask,
            },
            batch_size=coords.shape[0],
        )

        # flow matching loss
        loss_dict = self.flow_model(latent_batch)
        loss = loss_dict["loss"]

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.flow_model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ae_ckpt", required=True, help="DinoAELitModule checkpoint")
    p.add_argument("--data_dir", required=True, help="train .pt file")
    p.add_argument("--val_data_dir", required=True, help="val .pt file")
    p.add_argument("--lmdb_dir", default="./lmdb_cache")
    p.add_argument("--output_dir", default="./outputs/dino_flow")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--resume_from", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    train_ds = UnconditionalLMDBDataset(
        data_dir=args.data_dir,
        split="train",
        lmdb_dir=os.path.join(args.lmdb_dir, "train"),
    )
    val_ds = UnconditionalLMDBDataset(
        data_dir=args.val_data_dir,
        split="val",
        lmdb_dir=os.path.join(args.lmdb_dir, "val"),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = DinoFlowLitModule(
        ae_ckpt=args.ae_ckpt,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
    )
    logger = CSVLogger(args.output_dir, name="logs")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        callbacks=[ckpt_cb],
        logger=logger,
        gradient_clip_val=1.0,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
