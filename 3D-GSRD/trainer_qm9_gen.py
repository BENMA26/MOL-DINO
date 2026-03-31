import math
import argparse
import copy
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
from flow_matching.models.components.losses import InterDistancesLoss
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
        self.last_step = -1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self, it):
        if it < self.warmup_iters:
            return self.init_lr * it / max(self.warmup_iters, 1)
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / max(self.lr_decay_iters - self.warmup_iters, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step=None):
        if cur_step is None:
            cur_step = self.last_step + 1
        self.last_step = cur_step
        lr = self.get_lr(cur_step)
        new_lrs = []
        for param_group in self.optimizer.param_groups:
            lr_scale = float(param_group.get('lr_scale', 1.0))
            group_lr = lr * lr_scale
            param_group['lr'] = group_lr
            new_lrs.append(group_lr)
        self._last_lr = new_lrs
        return lr

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'max_iters': self.max_iters,
            'min_lr': self.min_lr,
            'init_lr': self.init_lr,
            'warmup_iters': self.warmup_iters,
            'warmup_start_lr': self.warmup_start_lr,
            'lr_decay_iters': self.lr_decay_iters,
            'last_step': self.last_step,
            '_last_lr': self._last_lr,
        }

    def load_state_dict(self, state_dict):
        self.max_iters = state_dict.get('max_iters', self.max_iters)
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.init_lr = state_dict.get('init_lr', self.init_lr)
        self.warmup_iters = state_dict.get('warmup_iters', self.warmup_iters)
        self.warmup_start_lr = state_dict.get('warmup_start_lr', self.warmup_start_lr)
        self.lr_decay_iters = state_dict.get('lr_decay_iters', self.lr_decay_iters)
        self.last_step = state_dict.get('last_step', self.last_step)
        self._last_lr = state_dict.get('_last_lr', self._last_lr)


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

        # 两阶段训练逻辑 (UNILIP-style)
        if args.stage == 1:
            # Stage 1: 固定编码器，只训练解码器
            encoder.requires_grad_(False)
            print("Stage 1: Encoder frozen, training decoder only.")
        elif args.stage == 2:
            # Stage 2: 微调编码器和解码器
            encoder.requires_grad_(True)
            print("Stage 2: Finetuning both encoder and decoder.")

        # 兼容旧的 freeze_encoder 参数
        if args.freeze_encoder:
            encoder.requires_grad_(False)
            print("Encoder frozen (via --freeze_encoder flag).")

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

        interdist_loss = None
        interdist_threshold = None
        if args.geometry_dist_threshold > 0:
            interdist_threshold = args.geometry_dist_threshold
        if args.geometry_dist_loss_weight > 0:
            interdist_loss = InterDistancesLoss(
                distance_threshold=interdist_threshold,
                sqrd=False,
                key='coords',
                key_pad_mask='padding_mask',
            )

        # ── flow model ────────────────────────────────────────
        self.flow_model = FlowMatchingModel(
            net=net,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            time_distribution=args.time_distribution,
            time_alpha_factor=args.time_alpha_factor,
            interdist_loss=interdist_loss,
            interdist_loss_weight=args.geometry_dist_loss_weight,
            chirality_loss_weight=args.chirality_loss_weight,
            chirality_eps=args.chirality_eps,
            num_random_augmentations=None,
            sample_schedule=args.sample_schedule,
        )
        # Stage-2 ablation: optional latent noise at encoder output.
        if args.stage == 2 and args.latent_noise_std > 0:
            self.flow_model.net.latent_noise_std = args.latent_noise_std
            print(f"Stage 2: latent noise enabled (std={args.latent_noise_std}).")

        # Stage-2 ablation: optional self-distillation.
        self.use_self_distill = args.stage == 2 and args.distill_weight > 0
        self.distill_decay = args.distill_decay
        self.distill_min_ratio = float(args.distill_min_ratio)
        self.teacher_encoder = None

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

    def _encode_dense_with_encoder(self, encoder, batch):
        """Encode a PyG batch with a specific encoder and convert to dense nodes."""
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            edge_feature = batch.edge_attr
        else:
            edge_dim = encoder.edge_embedding[0].in_features - 1
            edge_feature = torch.zeros(
                batch.edge_index.shape[1],
                edge_dim,
                device=batch.pos.device,
                dtype=batch.pos.dtype,
            )

        h, _ = encoder(
            data=batch,
            node_feature=batch.x,
            edge_index=batch.edge_index,
            edge_feature=edge_feature,
            position=batch.pos,
            pos_mask=None,
            return_cls=False,
            return_node_rep=True,
        )
        h_dense, mask = to_dense_batch(h, batch.batch)
        padding_mask = ~mask
        return h_dense, padding_mask

    def _maybe_init_teacher_encoder(self):
        if not self.use_self_distill:
            return
        if self.teacher_encoder is not None:
            return
        self.teacher_encoder = copy.deepcopy(self.flow_model.net.encoder)
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()
        print("Stage 2: initialized self-distillation teacher encoder.")

    def _distill_decay_ratio(self):
        mode = self.distill_decay
        if mode in {'none', 'None'}:
            return 1.0

        min_ratio = min(max(self.distill_min_ratio, 0.0), 1.0)
        progress = 0.0

        trainer = getattr(self, 'trainer', None)
        if trainer is not None:
            total_steps = int(getattr(trainer, 'estimated_stepping_batches', 0) or 0)
            if total_steps > 0:
                progress = min(max(self.global_step / float(total_steps), 0.0), 1.0)
            else:
                max_epochs = int(getattr(self.args, 'max_epochs', 0) or 0)
                if max_epochs > 0:
                    progress = min(max(self.current_epoch / float(max_epochs), 0.0), 1.0)

        if mode == 'linear':
            ratio = 1.0 - (1.0 - min_ratio) * progress
        elif mode == 'cosine':
            ratio = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            ratio = 1.0
        return min(max(ratio, min_ratio), 1.0)

    def _compute_distill_loss(self, batch):
        if not self.use_self_distill:
            return None
        self._maybe_init_teacher_encoder()

        with torch.no_grad():
            h_teacher, pad_teacher = self._encode_dense_with_encoder(self.teacher_encoder, batch)
        h_student, pad_student = self._encode_dense_with_encoder(self.flow_model.net.encoder, batch)

        valid_mask = (~pad_student).unsqueeze(-1).to(h_student.dtype)
        denom = valid_mask.sum().clamp_min(1.0)
        distill_loss = ((h_student - h_teacher) ** 2 * valid_mask).sum() / denom
        return distill_loss

    # ── lightning hooks ───────────────────────────────────────

    def on_fit_start(self):
        # 从 datamodule 获取原子数直方图（如果有）
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'num_atoms_histogram'):
            self._data_stats['num_atoms_histogram'] = self.trainer.datamodule.num_atoms_histogram
        self.flow_model.set_data_stats(self._data_stats)
        self._maybe_init_teacher_encoder()
        if self.use_self_distill:
            print(
                "Stage 2 distill config: "
                f"weight={self.args.distill_weight}, decay={self.distill_decay}, min_ratio={self.distill_min_ratio}"
            )

    def training_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        flow_loss, _ = self.flow_model(flow_batch, compute_stats=False)
        loss = flow_loss

        if self.use_self_distill:
            distill_loss = self._compute_distill_loss(batch)
            distill_weight_eff = self.args.distill_weight * self._distill_decay_ratio()
            loss = loss + distill_weight_eff * distill_loss
            self.log(
                'train/distill_loss',
                distill_loss,
                batch_size=self.args.batch_size,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                'train/flow_loss',
                flow_loss,
                batch_size=self.args.batch_size,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                'train/distill_weight_eff',
                distill_weight_eff,
                batch_size=self.args.batch_size,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

        self.log('train/loss', loss, batch_size=self.args.batch_size,
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._set_current_data(batch)
        #print("batch.batch shape:", batch.batch.shape)  # 加这行
        #print("batch.pos shape:", batch.pos.shape)      # 加这行
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
        if self.args.stage == 2 and self.args.encoder_lr_ratio != 1.0:
            encoder_params = [
                p for p in self.flow_model.net.encoder.parameters() if p.requires_grad
            ]
            encoder_ids = {id(p) for p in encoder_params}
            other_params = [
                p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids
            ]
            param_groups = []
            if other_params:
                param_groups.append(
                    {'params': other_params, 'lr': self.args.init_lr, 'lr_scale': 1.0}
                )
            if encoder_params:
                param_groups.append(
                    {
                        'params': encoder_params,
                        'lr': self.args.init_lr * self.args.encoder_lr_ratio,
                        'lr_scale': self.args.encoder_lr_ratio,
                    }
                )
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.args.weight_decay,
            )
            print(
                f"Stage 2: using encoder_lr_ratio={self.args.encoder_lr_ratio} "
                f"(encoder lr={self.args.init_lr * self.args.encoder_lr_ratio:.2e}, "
                f"others lr={self.args.init_lr:.2e})."
            )
        else:
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

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, LinearWarmupCosineLRSchedulerV2):
            scheduler.step(self.global_step)
        elif metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)


def main(args):
    L.seed_everything(args.seed, workers=True)

    dm = QM9DM(args)
    dm.setup('fit')

    prior_model = Atomref(dataset=dm.dataset) if args.prior_model else None

    trainer_model = FlowMatchingTrainer(args, prior_model=prior_model)

    # Stage 2: 如果从 Stage 1 checkpoint 恢复，只加载模型权重
    if args.stage == 2 and args.ckpt_path:
        print(f"Stage 2: Loading model weights from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        trainer_model.load_state_dict(ckpt['state_dict'], strict=False)
        print("Stage 2: Model weights loaded successfully (optimizer state skipped)")
        # 清空 ckpt_path，避免 Lightning 再次尝试加载
        args.ckpt_path = None

    if args.disable_compile or (args.stage == 2 and args.distill_weight > 0):
        if args.stage == 2 and args.distill_weight > 0 and not args.disable_compile:
            print("Compile disabled because self-distillation is enabled in stage 2.")
        pass
    else:
        trainer_model.flow_model.net.encoder = torch.compile(
            trainer_model.flow_model.net.encoder,
            dynamic=True, fullgraph=False,
        )

    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    # 配置 DDP strategy（Stage 1 需要 find_unused_parameters=True）
    from lightning.pytorch.strategies import DDPStrategy
    if isinstance(device_cast(args.devices), list) and len(device_cast(args.devices)) > 1:
        # 多 GPU 训练，使用 DDP
        if args.stage == 1:
            # Stage 1: encoder 冻结，需要 find_unused_parameters=True
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            # Stage 2: 所有参数都参与训练
            strategy = DDPStrategy(find_unused_parameters=False)
    else:
        # 单 GPU 训练
        strategy = 'auto'

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=device_cast(args.devices),
        precision=args.precision,
        strategy=strategy,
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

    # ── Two-stage training (UNILIP-style) ────────────────────
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Training stage: 1=freeze encoder, 2=finetune encoder')
    parser.add_argument('--stage1_ckpt', type=str, default=None,
                        help='Checkpoint from stage 1 to resume stage 2 training')
    parser.add_argument('--distill_weight', type=float, default=0.0,
                        help='Stage-2 self-distillation loss weight (0 disables distillation)')
    parser.add_argument('--distill_decay', type=str, default='none',
                        choices=['none', 'linear', 'cosine'],
                        help='Decay schedule for distillation weight during Stage-2 finetuning')
    parser.add_argument('--distill_min_ratio', type=float, default=0.0,
                        help='Minimum ratio for decayed distill weight (effective weight = distill_weight * ratio)')
    parser.add_argument('--encoder_lr_ratio', type=float, default=1.0,
                        help='Stage-2 encoder lr ratio relative to init_lr (e.g., 0.1)')
    parser.add_argument('--latent_noise_std', type=float, default=0.0,
                        help='Stage-2 latent gaussian noise std at encoder output (train only)')
    parser.add_argument('--geometry_dist_loss_weight', type=float, default=0.0,
                        help='Optional geometry prior: inter-atomic distance loss weight')
    parser.add_argument('--geometry_dist_threshold', type=float, default=0.0,
                        help='Optional threshold for geometry distance loss (<=0 disables threshold)')
    parser.add_argument('--chirality_loss_weight', type=float, default=0.0,
                        help='Optional local chirality consistency loss weight')
    parser.add_argument('--chirality_eps', type=float, default=1e-4,
                        help='Signed-volume magnitude cutoff for chirality loss')

    # ── flow matching ─────────────────────────────────────────
    parser.add_argument('--time_distribution', type=str, default='uniform',
                        choices=['uniform', 'beta', 'histogram'])
    parser.add_argument('--time_alpha_factor', type=float, default=2.0)
    parser.add_argument('--sample_schedule', type=str, default='linear',
                        choices=['linear', 'power', 'log'])

    args = parser.parse_args()
    main(args)
