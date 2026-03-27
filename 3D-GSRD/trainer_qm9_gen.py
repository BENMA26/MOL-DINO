import math
import argparse
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from tensordict import TensorDict
from torch_geometric.utils import to_dense_batch

from model.retrans import RelaTransEncoder
from flow_matching.models.flow_matching_model import FlowMatchingModel
from flow_matching.models.combined_net import CombinedNet
from flow_matching.models.components.dae_decoder import DAEDecoder
from flow_matching.flow.interpolate import CenteredMetricInterpolant, DiscreteInterpolant
from data_provider.qm9_dm import QM9DM
from training_utils import custom_callbacks, load_encoder_params, print_args, device_cast
from atomref import Atomref

torch.set_float32_matmul_precision('high')


class LinearWarmupCosineLRSchedulerV2:
    def __init__(self, optimizer, max_iters, min_lr, init_lr,
                 warmup_iters=0, warmup_start_lr=-1, **kwargs):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.lr_decay_iters = max_iters

    def get_lr(self, it):
        if it < self.warmup_iters:
            return self.init_lr * it / max(self.warmup_iters, 1)
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class FlowMatchingTrainer(L.LightningModule):

    def __init__(self, args, prior_model=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['prior_model'])

        # ── encoder ───────────────────────────────────────────
        encoder = RelaTransEncoder(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_blocks=args.encoder_blocks,
            prior_model=prior_model,
            args=args,
        )

        if args.encoder_ckpt:
            encoder = load_encoder_params(encoder, args.encoder_ckpt)
            print(f"Loaded encoder weights from {args.encoder_ckpt}")

        if args.freeze_encoder:
            encoder.requires_grad_(False)
            print("Encoder frozen.")

        # ── decoder ───────────────────────────────────────────
        decoder = DAEDecoder(
            spatial_dim=3,
            atom_dim=args.atom_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.n_heads,
            num_layers=args.decoder_blocks,
            add_sinusoid_posenc=True,
            concat_combine_input=False,
            cross_attention=False,
            implementation='pytorch',
        )

        # ── combined net ──────────────────────────────────────
        net = CombinedNet(
            encoder=encoder,
            decoder=decoder,
            hidden_dim=args.hidden_dim,
            kl_dim=args.kl_dim,
            kl_weight=args.kl_weight,
        )

        # ── interpolants ──────────────────────────────────────
        coords_interpolant = CenteredMetricInterpolant(
            key='coords',
            key_pad_mask='padding_mask',
            centered=True,
            noise_scale=1.0,
        )
        atomics_interpolant = DiscreteInterpolant(
            key='atomics',
            key_pad_mask='padding_mask',
        )

        # ── flow model ────────────────────────────────────────
        self.flow_model = FlowMatchingModel(
            net=net,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            time_distribution=args.time_distribution,
            time_alpha_factor=args.time_alpha_factor,
            interdist_loss=None,
            num_random_augmentations=None,
            sample_schedule=args.sample_schedule,
        )

        # data_stats 在 on_fit_start 里注入
        self._data_stats = {
            'spatial_dim': 3,
            'atom_dim': args.atom_dim,
            'max_num_atoms': args.max_num_atoms,
            'num_atoms_histogram': {},
        }

    # ── helpers ───────────────────────────────────────────────

    def _set_current_data(self, batch):
        self.flow_model.net._current_data = batch

    def _to_flow_batch(self, batch):
        coords_dense, mask = to_dense_batch(batch.pos, batch.batch)   # [B, N, 3]
        padding_mask = ~mask                                            # True = padding

        # data.x 前 atom_dim 列是原子类型 one-hot
        atom_types = batch.x[:, :self.args.atom_dim].argmax(dim=-1).long()
        atomics_packed = F.one_hot(atom_types, num_classes=self.args.atom_dim).float()
        atomics_dense, _ = to_dense_batch(atomics_packed, batch.batch) # [B, N, atom_dim]

        B = coords_dense.shape[0]
        return TensorDict(
            {'coords': coords_dense, 'atomics': atomics_dense, 'padding_mask': padding_mask},
            batch_size=B,
        ).to(batch.pos.device)

    # ── lightning hooks ───────────────────────────────────────

    def on_fit_start(self):
        # 从 datamodule 获取原子数直方图（如果有）
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'num_atoms_histogram'):
            self._data_stats['num_atoms_histogram'] = self.trainer.datamodule.num_atoms_histogram
        self.flow_model.set_data_stats(self._data_stats)

    def training_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, _ = self.flow_model(flow_batch, compute_stats=False)
        self.log('train/loss', loss, batch_size=self.args.batch_size,
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, stats = self.flow_model(flow_batch, compute_stats=True)
        self.log('val/loss', loss, batch_size=self.args.inference_batch_size, prog_bar=True)
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                self.log(f'val/{k}', v, batch_size=self.args.inference_batch_size)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(f'val/{k}', v.item(), batch_size=self.args.inference_batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, stats = self.flow_model(flow_batch, compute_stats=True)
        self.log('test/loss', loss, batch_size=self.args.inference_batch_size)
        if 'rmsd' in stats:
            self.log('test/rmsd', stats['rmsd'], batch_size=self.args.inference_batch_size)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.args.init_lr, weight_decay=self.args.weight_decay
        )

        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.trainer.fit_loop.setup_data()
            max_iters = (
                self.args.max_epochs
                * len(self.trainer.train_dataloader)
                // self.args.accumulate_grad_batches
            )
            scheduler = LinearWarmupCosineLRSchedulerV2(
                optimizer, max_iters,
                self.args.min_lr, self.args.init_lr,
                self.args.warmup_steps, self.args.warmup_lr,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
            }
        elif self.args.scheduler in {'none', 'None'}:
            return optimizer
        else:
            raise NotImplementedError(f"Unknown scheduler: {self.args.scheduler}")


def main(args):
    L.seed_everything(args.seed, workers=True)

    dm = QM9DM(args)
    dm.setup('fit')

    prior_model = Atomref(dataset=dm.dataset) if args.prior_model else None

    trainer_model = FlowMatchingTrainer(args, prior_model=prior_model)

    if args.disable_compile:
        pass
    else:
        trainer_model.flow_model.net.encoder = torch.compile(
            trainer_model.flow_model.net.encoder,
            dynamic=True, fullgraph=False,
        )

    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=device_cast(args.devices),
        precision=args.precision,
        logger=logger,
        callbacks=custom_callbacks(args),
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        detect_anomaly=args.detect_anomaly,
    )

    print_args(parser, args)

    if args.test_only:
        trainer.test(trainer_model, datamodule=dm, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(trainer_model, datamodule=dm, ckpt_path=args.ckpt_path)
        trainer.test(trainer_model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = QM9DM.add_model_specific_args(parser)

    # ── identity ──────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filename', type=str, default='flow_ae_qm9')

    # ── hardware ──────────────────────────────────────────────
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--disable_compile', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    # ── training schedule ─────────────────────────────────────
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=5_000_000)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--save_every_n_epochs', type=int, default=50)
    parser.add_argument('--test_every_n_epochs', type=int, default=None)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Resume training or test from this checkpoint')

    # ── optimizer ─────────────────────────────────────────────
    parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr')
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--weight_decay', type=float, default=1e-8)

    # ── encoder ───────────────────────────────────────────────
    parser.add_argument('--node_dim', type=int, default=63)
    parser.add_argument('--edge_dim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--encoder_blocks', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--pair_update', action='store_true', default=False)
    parser.add_argument('--trans_version', type=str, default='v6')
    parser.add_argument('--attn_activation', type=str, default='silu')
    parser.add_argument('--prior_model', action='store_true', default=False)
    parser.add_argument('--use_cls_token', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--dataset_arg', type=str, default='homo')
    parser.add_argument('--delta', type=int, default=1000)
    parser.add_argument('--encoder_ckpt', type=str, default='',
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze encoder weights during training')

    # ── decoder ───────────────────────────────────────────────
    parser.add_argument('--decoder_blocks', type=int, default=8)
    parser.add_argument('--atom_dim', type=int, default=5,
                        help='Number of atom types (5 for QM9: H C N O F)')
    parser.add_argument('--max_num_atoms', type=int, default=29,
                        help='Max atoms per molecule in QM9')

    # ── VAE bottleneck ────────────────────────────────────────
    parser.add_argument('--kl_dim', type=int, default=6)
    parser.add_argument('--kl_weight', type=float, default=1e-6)

    # ── flow matching ─────────────────────────────────────────
    parser.add_argument('--time_distribution', type=str, default='uniform',
                        choices=['uniform', 'beta', 'histogram'])
    parser.add_argument('--time_alpha_factor', type=float, default=2.0)
    parser.add_argument('--sample_schedule', type=str, default='linear',
                        choices=['linear', 'power', 'log'])

    args = parser.parse_args()
    main(args)
