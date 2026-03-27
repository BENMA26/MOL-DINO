"""
FlowMatchingTrainer: Lightning Module，整合 CombinedNet 和 FlowMatchingModel。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from tensordict import TensorDict
from torch_geometric.utils import to_dense_batch

from flow_matching.models.flow_matching_model import FlowMatchingModel
from flow_matching.models.combined_net import CombinedNet
from flow_matching.models.components.dae_decoder import DAEDecoder
from flow_matching.flow.interpolate import CenteredMetricInterpolant, DiscreteInterpolant
from model.retrans import RelaTransEncoder


class LinearWarmupCosineLRSchedulerV2:
    def __init__(self, optimizer, max_iters, min_lr, init_lr, warmup_iters=0, warmup_start_lr=-1, **kwargs):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.lr_decay_iters = max_iters

    def get_lr(self, it):
        if it < self.warmup_iters:
            return self.init_lr * it / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class FlowMatchingTrainer(L.LightningModule):

    def __init__(self, args, prior_model=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['prior_model'])

        # ── 构建 encoder ──────────────────────────────────────
        encoder = RelaTransEncoder(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_blocks=args.encoder_blocks,
            prior_model=prior_model,
            args=args,
        )

        # 加载预训练权重（如果有）
        if hasattr(args, 'encoder_ckpt') and args.encoder_ckpt:
            ckpt = torch.load(args.encoder_ckpt, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            encoder_state = {
                k.replace('model.encoder.', '').replace('encoder.', ''): v
                for k, v in state_dict.items()
                if 'encoder' in k
            }
            missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
            print(f"Encoder loaded. Missing: {missing}, Unexpected: {unexpected}")

        if getattr(args, 'freeze_encoder', False):
            encoder.requires_grad_(False)
            print("Encoder frozen.")

        # ── 构建 decoder ──────────────────────────────────────
        decoder = DAEDecoder(
            spatial_dim=getattr(args, 'spatial_dim', 3),
            atom_dim=args.atom_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.n_heads,
            num_layers=args.decoder_blocks,
            add_sinusoid_posenc=True,
            concat_combine_input=False,
            cross_attention=False,
            implementation='pytorch',
        )

        # ── 组合成 CombinedNet ────────────────────────────────
        net = CombinedNet(
            encoder=encoder,
            decoder=decoder,
            hidden_dim=args.hidden_dim,
            kl_dim=getattr(args, 'kl_dim', 6),
            kl_weight=getattr(args, 'kl_weight', 1e-6),
        )

        # ── 构建 Interpolant ──────────────────────────────────
        coords_interpolant = CenteredMetricInterpolant(
            key="coords",
            key_pad_mask="padding_mask",
            centered=True,
            noise_scale=1.0,
        )
        atomics_interpolant = DiscreteInterpolant(
            key="atomics",
            key_pad_mask="padding_mask",
        )

        # ── 构建 FlowMatchingModel ────────────────────────────
        self.flow_model = FlowMatchingModel(
            net=net,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            time_distribution=getattr(args, 'time_distribution', 'uniform'),
            time_alpha_factor=getattr(args, 'time_alpha_factor', 2.0),
            interdist_loss=None,
            num_random_augmentations=getattr(args, 'num_random_augmentations', None),
            sample_schedule=getattr(args, 'sample_schedule', 'linear'),
        )

        self._data_stats = {
            'spatial_dim': getattr(args, 'spatial_dim', 3),
            'atom_dim': args.atom_dim,
            'max_num_atoms': args.max_num_atoms,
            'num_atoms_histogram': {},
        }

        self._cur_step = 0

    # ── 接口适配 ──────────────────────────────────────────────

    def _set_current_data(self, batch):
        """把 PyG batch 注入 CombinedNet，供 encode_z/forward 使用。"""
        self.flow_model.net._current_data = batch

    def _to_flow_batch(self, batch):
        """把 PyG batch 转换为 FlowMatchingModel 期望的 TensorDict。"""
        coords_dense, mask = to_dense_batch(batch.pos, batch.batch)  # [B, N, 3]
        padding_mask = ~mask                                           # [B, N]

        # data.x 前 atom_dim 列是原子类型 one-hot（来自 featurization.py 的 x1）
        # 用 argmax 还原为整数 index，再转回 one-hot（统一格式）
        atom_types = batch.x[:, :self.args.atom_dim].argmax(dim=-1).long()
        atomics_packed = F.one_hot(
            atom_types, num_classes=self.args.atom_dim
        ).float()                                                      # [total_nodes, atom_dim]
        atomics_dense, _ = to_dense_batch(atomics_packed, batch.batch)  # [B, N, atom_dim]

        B = coords_dense.shape[0]
        return TensorDict(
            {
                'coords': coords_dense,
                'atomics': atomics_dense,
                'padding_mask': padding_mask,
            },
            batch_size=B,
        ).to(batch.pos.device)

    # ── Lightning 标准接口 ────────────────────────────────────

    def on_fit_start(self):
        self.flow_model.set_data_stats(self._data_stats)

    def training_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, stats = self.flow_model(flow_batch, compute_stats=False)
        self.log('train/loss', loss, batch_size=self.args.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, stats = self.flow_model(flow_batch, compute_stats=True)
        self.log('val/loss', loss, batch_size=self.args.batch_size, prog_bar=True)
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                self.log(f'val/{k}', v, batch_size=self.args.batch_size)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(f'val/{k}', v.item(), batch_size=self.args.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        self._set_current_data(batch)
        flow_batch = self._to_flow_batch(batch)
        loss, stats = self.flow_model(flow_batch, compute_stats=True)
        self.log('test/loss', loss, batch_size=self.args.batch_size)
        if 'rmsd' in stats:
            self.log('test/rmsd', stats['rmsd'], batch_size=self.args.batch_size)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.args.init_lr,
            weight_decay=self.args.weight_decay,
        )

        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.trainer.fit_loop.setup_data()
            max_iters = (
                self.args.max_epochs
                * len(self.trainer.train_dataloader)
                // self.args.accumulate_grad_batches
            )
            self._scheduler = LinearWarmupCosineLRSchedulerV2(
                optimizer,
                max_iters,
                self.args.min_lr,
                self.args.init_lr,
                self.args.warmup_steps,
                self.args.warmup_lr,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': self._scheduler, 'interval': 'step'},
            }
        elif self.args.scheduler in {'none', 'None'}:
            return optimizer
        else:
            raise NotImplementedError(f"Scheduler '{self.args.scheduler}' not implemented")
