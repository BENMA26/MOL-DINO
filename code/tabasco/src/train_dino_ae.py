"""
Stage 1 training: freeze DINO encoder, train decoder only.

Usage:
    python train_dino_ae.py \
        --gsrd_checkpoint /path/to/pretrain.ckpt \
        --data_dir /path/to/crossdocked_train.pt \
        --val_data_dir /path/to/crossdocked_val.pt \
        --output_dir ./outputs/dino_ae_stage1 \
        --freeze_encoder \
        --max_epochs 100 \
        --batch_size 64

Stage 2 (unfreeze encoder):
    python train_dino_ae.py \
        --gsrd_checkpoint /path/to/pretrain.ckpt \
        --data_dir /path/to/crossdocked_train.pt \
        --val_data_dir /path/to/crossdocked_val.pt \
        --output_dir ./outputs/dino_ae_stage2 \
        --resume_from ./outputs/dino_ae_stage1/best.ckpt \
        --no_freeze_encoder \
        --max_epochs 50 \
        --batch_size 32
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

# ── make tabasco importable ───────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_tabasco_src = os.path.join(_here, "../../tabasco/src")
if _tabasco_src not in sys.path:
    sys.path.insert(0, _tabasco_src)

from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from tabasco.data.lmdb_datamodule import LMDBDataModule
from tabasco.models.flow_model import FlowMatchingModel
from tabasco.flow.interpolate import Interpolant
from tabasco.models.dino_encoder import DinoEncoderModule


# ── build a minimal gsrd_args Namespace ──────────────────────────────────────

def make_gsrd_args(hidden_dim=256, n_heads=8, encoder_blocks=6):
    """Minimal args Namespace matching RelaTransEncoder constructor."""
    import argparse
    a = argparse.Namespace()
    # node / edge feature dims from featurization.py (merge_types = 22 atom types + extra feats)
    a.node_dim = 22 + 3        # atom one-hot (22) + 3D coords appended inside encoder
    a.edge_dim = 1             # placeholder (distance-only edges)
    a.hidden_dim = hidden_dim
    a.n_heads = n_heads
    a.encoder_blocks = encoder_blocks
    a.dropout = 0.0
    a.pair_update = True
    a.trans_version = "v6"
    a.attn_activation = "silu"
    a.dataset = "pcqm4mv2"     # triggers cutoff in TransLayerOptimV6
    a.dataset_arg = None
    a.use_cls_token = False
    a.cls_distance = 1.0
    a.pos_mask = False
    a.denoising = False
    return a


# ── Lightning module ──────────────────────────────────────────────────────────

class DinoAELitModule(L.LightningModule):
    def __init__(
        self,
        gsrd_checkpoint: str,
        gsrd_hidden_dim: int = 256,
        hidden_dim: int = 256,
        kl_dim: int = 6,
        kl_weight: float = 1e-6,
        atom_dim: int = 16,
        spatial_dim: int = 3,
        num_heads: int = 8,
        num_layers: int = 8,
        freeze_encoder: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()

        gsrd_args = make_gsrd_args(
            hidden_dim=gsrd_hidden_dim,
            n_heads=num_heads,
            encoder_blocks=6,
        )

        self.net = DinoEncoderModule(
            gsrd_args=gsrd_args,
            gsrd_checkpoint_path=gsrd_checkpoint,
            kl_dim=kl_dim,
            kl_weight=kl_weight,
            spatial_dim=spatial_dim,
            atom_dim=atom_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            freeze_encoder=freeze_encoder,
        )

        # simple reconstruction losses
        self.coord_loss = torch.nn.MSELoss()
        self.atom_loss = torch.nn.CrossEntropyLoss()

    def _step(self, batch):
        coords = batch["coords"]          # (B, N, 3)
        atomics = batch["atomics"]        # (B, N, A)
        padding_mask = batch["padding_mask"]  # (B, N)

        # use t=0 for pure reconstruction (no flow noise)
        B = coords.shape[0]
        t = torch.zeros(B, device=self.device)

        pred_coords, pred_atom_logits, kl_loss = self.net(
            coords, atomics, padding_mask,
            coords, atomics, t,
        )

        real = ~padding_mask  # (B, N)
        c_loss = self.coord_loss(pred_coords[real], coords[real])
        a_loss = self.atom_loss(
            pred_atom_logits[real],
            atomics[real].argmax(dim=-1),
        )
        loss = c_loss + a_loss + kl_loss
        return loss, c_loss, a_loss, kl_loss

    def training_step(self, batch, _):
        loss, c, a, kl = self._step(batch)
        self.log_dict({"train/loss": loss, "train/coord": c, "train/atom": a, "train/kl": kl},
                      prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, c, a, kl = self._step(batch)
        self.log_dict({"val/loss": loss, "val/coord": c, "val/atom": a, "val/kl": kl},
                      on_epoch=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gsrd_checkpoint", required=True)
    p.add_argument("--data_dir", required=True, help="path to train .pt mol list")
    p.add_argument("--val_data_dir", required=True, help="path to val .pt mol list")
    p.add_argument("--lmdb_dir", default="./lmdb_cache")
    p.add_argument("--output_dir", default="./outputs/dino_ae")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--freeze_encoder", action="store_true", default=True)
    p.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--gsrd_hidden_dim", type=int, default=256)
    p.add_argument("--kl_dim", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: 2 epochs, 8 samples, no GPU precision tricks")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── debug mode ────────────────────────────────────────────────────────────
    if args.debug:
        print("=" * 50)
        print("DEBUG MODE: 2 epochs, 8 samples, CPU, no precision tricks")
        print("=" * 50)
        args.max_epochs = 2
        args.batch_size = 2
        args.num_workers = 0

    # ── data ─────────────────────────────────────────────────────────────────
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

    # debug: 只用少量样本
    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(8, len(train_ds))))
        val_ds = Subset(val_ds, range(min(8, len(val_ds))))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    # infer atom_dim from dataset
    sample = train_ds[0]
    atom_dim = sample["atomics"].shape[-1]

    model = DinoAELitModule(
        gsrd_checkpoint=args.gsrd_checkpoint,
        gsrd_hidden_dim=args.gsrd_hidden_dim,
        hidden_dim=args.hidden_dim,
        kl_dim=args.kl_dim,
        atom_dim=atom_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        freeze_encoder=args.freeze_encoder,
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
        accelerator="gpu" if (torch.cuda.is_available() and not args.debug) else "cpu",
        devices=args.devices,
        callbacks=[ckpt_cb],
        logger=logger,
        gradient_clip_val=1.0,
        precision="bf16-mixed" if (torch.cuda.is_available() and not args.debug) else 32,
        fast_dev_run=args.debug,
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
