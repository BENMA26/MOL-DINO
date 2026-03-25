import math
import copy
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from model.retrans import RelaTransEncoder
from model.autoencoder import AutoEncoder
import argparse
from torch_geometric.data import Data, Batch
from data_provider.pcqm4mv2_dm import PCQM4MV2DM
from training_utils import custom_callbacks,load_encoder_params
from atomref import Atomref

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

class LinearWarmupCosineLRSchedulerV2:
    def __init__(
        self,
        optimizer,
        max_iters,
        min_lr,
        init_lr,
        warmup_iters=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.lr_decay_iters = max_iters

    def get_lr(self, it):
        # 1) linear warmup for warmup_steps steps
        if it < self.warmup_iters:
            return self.init_lr * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
     
    
class AutoEncoderTrainer(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = nn.MSELoss()
        self.scheduler = None
        self.graph_emb_dim = args.hidden_dim * (2 if args.use_cls_token else 1)
        self.use_local_global_distill = args.local_global_distill
        if self.use_local_global_distill:
            self.student_projector = nn.Sequential(
                nn.Linear(self.graph_emb_dim, args.hidden_dim),
                nn.GELU(),
                nn.Linear(args.hidden_dim, args.local_proj_dim),
            )
            self.student_prototypes = nn.Linear(args.local_proj_dim, args.local_num_prototypes, bias=False)
            self.teacher_encoder = copy.deepcopy(self.model.encoder)
            self.teacher_projector = copy.deepcopy(self.student_projector)
            self.teacher_prototypes = copy.deepcopy(self.student_prototypes)
            for module in (self.teacher_encoder, self.teacher_projector, self.teacher_prototypes):
                module.requires_grad_(False)
            self.register_buffer("teacher_center", torch.zeros(1, args.local_num_prototypes))
        self.save_hyperparameters(ignore='model')

    def forward(self, batch):
        if self.args.pos_mask and self.args.denoising:
            prediction, pred_noise, vec, _ = self.model(batch)
            target = batch.mask_coord_label
            loss_pos = self.criterion(pred_noise, batch.pos_target[~batch.pos_mask])
            excess_norm = F.relu(vec.norm(dim=-1) - self.args.delta)
            loss_vec_norm = 0.01 * excess_norm.mean()
            loss1 = self.criterion(prediction, target)
            return loss1, loss_pos, loss_vec_norm

        if not self.args.pos_mask and self.args.denoising:
            pred_noise, vec = self.model(batch)
            loss_pos = self.criterion(pred_noise, batch.pos_target)
            excess_norm = F.relu(vec.norm(dim=-1) - self.args.delta)
            loss_vec_norm = 0.01 * excess_norm.mean()
            return loss_pos, loss_vec_norm

        if self.args.pos_mask and not self.args.denoising:
            prediction, vec = self.model(batch)
            target = batch.mask_coord_label
            loss1 = self.criterion(prediction, target)
            excess_norm = F.relu(vec.norm(dim=-1) - self.args.delta)
            loss_vec_norm = 0.01 * excess_norm.mean()
            return loss1, loss_vec_norm

        raise NotImplementedError("Unsupported pretraining configuration")

    def _sample_local_nodes(self, pos):
        num_nodes = pos.size(0)
        if num_nodes <= self.args.local_min_nodes:
            return torch.arange(num_nodes, device=pos.device)

        center_idx = torch.randint(num_nodes, (1,), device=pos.device).item()
        dist = torch.norm(pos - pos[center_idx], dim=-1)
        keep_idx = torch.where(dist <= self.args.local_radius)[0]

        target_nodes = max(1, int(math.ceil(num_nodes * self.args.local_crop_ratio)))
        target_nodes = max(target_nodes, self.args.local_min_nodes)
        target_nodes = min(target_nodes, num_nodes)

        if keep_idx.numel() < target_nodes:
            keep_idx = torch.topk(dist, k=target_nodes, largest=False).indices
        if self.args.local_max_nodes > 0 and keep_idx.numel() > self.args.local_max_nodes:
            keep_idx = torch.topk(dist, k=self.args.local_max_nodes, largest=False).indices
        return keep_idx.sort().values

    def _build_local_crop_batch(self, batch):
        data_list = []
        edge_index = batch.edge_index
        num_graphs = int(batch.ptr.numel() - 1)

        for graph_idx in range(num_graphs):
            node_start = int(batch.ptr[graph_idx].item())
            node_end = int(batch.ptr[graph_idx + 1].item())
            num_nodes = node_end - node_start
            if num_nodes <= 0:
                continue

            node_slice = slice(node_start, node_end)
            keep_idx = self._sample_local_nodes(batch.pos[node_slice])
            keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch.x.device)
            keep_mask[keep_idx] = True

            edge_mask = (edge_index[0] >= node_start) & (edge_index[0] < node_end)
            local_edge_index = edge_index[:, edge_mask] - node_start
            local_edge_attr = batch.edge_attr[edge_mask]
            sub_edge_mask = keep_mask[local_edge_index[0]] & keep_mask[local_edge_index[1]]
            sub_edge_index = local_edge_index[:, sub_edge_mask]
            sub_edge_attr = local_edge_attr[sub_edge_mask]

            remap = torch.full((num_nodes,), -1, dtype=torch.long, device=batch.x.device)
            remap[keep_idx] = torch.arange(keep_idx.numel(), device=batch.x.device)
            sub_edge_index = remap[sub_edge_index]
            if sub_edge_index.numel() == 0:
                local_num_nodes = keep_idx.numel()
                diag = torch.arange(local_num_nodes, device=batch.x.device)
                sub_edge_index = torch.stack([diag, diag], dim=0)
                sub_edge_attr = batch.edge_attr.new_zeros((local_num_nodes, batch.edge_attr.size(-1)))

            pos_sub = batch.pos[node_slice][keep_idx]
            data_kwargs = {
                "x": batch.x[node_slice][keep_idx],
                "edge_index": sub_edge_index,
                "edge_attr": sub_edge_attr,
                "pos": pos_sub,
            }
            if hasattr(batch, "z"):
                data_kwargs["z"] = batch.z[node_slice][keep_idx]
            if hasattr(batch, "pos_mask"):
                pos_mask_sub = batch.pos_mask[node_slice][keep_idx]
                data_kwargs["pos_mask"] = pos_mask_sub
                data_kwargs["mask_coord_label"] = pos_sub[pos_mask_sub]
            if hasattr(batch, "pos_target"):
                data_kwargs["pos_target"] = batch.pos_target[node_slice][keep_idx]
            data_list.append(Data(**data_kwargs))

        local_batch = Batch.from_data_list(data_list)
        local_batch["max_seqlen"] = int((local_batch.ptr[1:] - local_batch.ptr[:-1]).max())
        return local_batch

    def _mean_pool(self, node_emb, node_batch, num_graphs):
        if node_emb.numel() == 0:
            return node_emb.new_zeros((num_graphs, self.args.hidden_dim))
        graph_emb = node_emb.new_zeros((num_graphs, node_emb.size(-1)))
        graph_cnt = node_emb.new_zeros((num_graphs, 1))
        graph_emb.index_add_(0, node_batch, node_emb)
        graph_cnt.index_add_(0, node_batch, node_emb.new_ones((node_batch.size(0), 1)))
        graph_cnt = graph_cnt.clamp_min(1.0)
        return graph_emb / graph_cnt

    def _encode_graph_embedding(self, batch, encoder, use_pos_mask):
        pos_mask = None
        if self.args.pos_mask and use_pos_mask and hasattr(batch, "pos_mask"):
            pos_mask = batch.pos_mask
        rep_out = encoder(
            data=batch,
            node_feature=batch.x,
            edge_index=batch.edge_index,
            edge_feature=batch.edge_attr,
            position=batch.pos,
            pos_mask=pos_mask,
            return_cls=self.args.use_cls_token,
        )
        node_batch = batch.batch if pos_mask is None else batch.batch[~pos_mask]
        num_graphs = int(batch.ptr.numel() - 1)
        if self.args.use_cls_token:
            rep, _, cls_rep = rep_out
            mean_rep = self._mean_pool(rep, node_batch, num_graphs)
            if cls_rep is None:
                return mean_rep
            return torch.cat([cls_rep, mean_rep], dim=-1)
        else:
            rep, _ = rep_out
            return self._mean_pool(rep, node_batch, num_graphs)

    @torch.no_grad()
    def _update_teacher_center(self, teacher_probs):
        batch_center = teacher_probs.mean(dim=0, keepdim=True)
        self.teacher_center.mul_(self.args.local_center_momentum).add_(
            batch_center, alpha=1.0 - self.args.local_center_momentum
        )

    @torch.no_grad()
    def _update_teacher(self):
        momentum = self.args.local_teacher_momentum
        for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.model.encoder.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_param, student_param in zip(self.teacher_projector.parameters(), self.student_projector.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_param, student_param in zip(self.teacher_prototypes.parameters(), self.student_prototypes.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    def _local_global_distill_loss(self, batch, update_center):
        local_batch = self._build_local_crop_batch(batch)
        local_emb = self._encode_graph_embedding(
            local_batch,
            self.model.encoder,
            use_pos_mask=self.args.local_student_use_mask,
        )
        student_feat = F.normalize(self.student_projector(local_emb), dim=-1)
        student_logits = self.student_prototypes(student_feat)

        with torch.no_grad():
            self.teacher_encoder.eval()
            self.teacher_projector.eval()
            self.teacher_prototypes.eval()
            global_emb = self._encode_graph_embedding(batch, self.teacher_encoder, use_pos_mask=False)
            teacher_feat = F.normalize(self.teacher_projector(global_emb), dim=-1)
            teacher_logits = self.teacher_prototypes(teacher_feat)
            teacher_probs = F.softmax(
                (teacher_logits - self.teacher_center) / self.args.local_teacher_temp,
                dim=-1,
            )
            if update_center:
                self._update_teacher_center(teacher_probs)

        student_log_probs = F.log_softmax(student_logits / self.args.local_student_temp, dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()

    def _compute_main_loss(self, batch, stage):
        if self.args.denoising and self.args.pos_mask:
            loss1, loss_pos, loss_vec_norm = self(batch)
            loss = loss1 + loss_pos * self.args.denoising_weight + loss_vec_norm
            self.log(f"{stage}_loss1", float(loss1), batch_size=self.args.batch_size)
            self.log(f"{stage}_loss_pos", float(loss_pos), batch_size=self.args.batch_size)
            self.log(f"{stage}_loss_vec_norm", float(loss_vec_norm), batch_size=self.args.batch_size)
            return loss

        if self.args.denoising and not self.args.pos_mask:
            loss_pos, loss_vec_norm = self(batch)
            loss = loss_pos + loss_vec_norm
            self.log(f"{stage}_loss_pos", float(loss_pos), batch_size=self.args.batch_size)
            self.log(f"{stage}_loss_vec_norm", float(loss_vec_norm), batch_size=self.args.batch_size)
            return loss

        loss1, loss_vec_norm = self(batch)
        loss = loss1 + loss_vec_norm
        self.log(f"{stage}_loss1", float(loss1), batch_size=self.args.batch_size)
        self.log(f"{stage}_loss_vec_norm", float(loss_vec_norm), batch_size=self.args.batch_size)
        return loss

    def on_train_epoch_start(self):
        if self.scheduler is not None and self.args.scheduler == 'cosine':
            self.scheduler.step()

    def on_train_step_start(self):
        if self.scheduler is not None and self.args.scheduler == 'frad_cosine':
            self.scheduler.step()

    def training_step(self, batch, batch_idx):
        loss = self._compute_main_loss(batch, stage="train")
        if self.use_local_global_distill:
            local_distill_loss = self._local_global_distill_loss(batch, update_center=True)
            loss = loss + self.args.local_distill_weight * local_distill_loss
            self.log("train_loss_local_distill", float(local_distill_loss), batch_size=self.args.batch_size)
        self.log("train_loss", float(loss), batch_size=self.args.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_main_loss(batch, stage="valid")
        if self.use_local_global_distill:
            local_distill_loss = self._local_global_distill_loss(batch, update_center=False)
            loss = loss + self.args.local_distill_weight * local_distill_loss
            self.log("valid_loss_local_distill", float(local_distill_loss), batch_size=self.args.batch_size)
        self.log("valid_loss", float(loss), batch_size=self.args.batch_size)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_local_global_distill:
            self._update_teacher()

    def test_step(self, batch, batch_idx):
        if self.args.denoising and self.args.pos_mask:
            prediction, _, _, _ = self.model(batch)
            target = batch.mask_coord_label
        elif self.args.denoising and not self.args.pos_mask:
            prediction, _ = self.model(batch)
            target = batch.pos_target
        else:
            prediction, _ = self.model(batch)
            target = batch.mask_coord_label
        mae = float(np.mean(np.square(prediction.cpu().detach().numpy() - target.cpu().detach().numpy())))
        self.log('test_MAE', mae, batch_size=self.args.batch_size)
        return mae

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = self.args.warmup_steps
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.args.init_lr,
            weight_decay=self.args.weight_decay
        )
        max_iters = self.args.max_epochs * len(self.trainer.train_dataloader) // self.args.accumulate_grad_batches
        assert max_iters > warmup_steps
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRSchedulerV2(optimizer, max_iters, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler in {'None', 'none'}:
            self.scheduler = None
        elif self.args.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(optimizer, self.args.max_epochs, eta_min=self.args.min_lr)
        elif self.args.scheduler == 'frad_cosine':
            self.scheduler = CosineAnnealingLR(optimizer, self.args.lr_cosine_length)
        else:
            raise NotImplementedError()
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.args.warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.args.warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.args.init_lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()
    

def main(args):
    L.seed_everything(args.seed,workers=True)
    dm = PCQM4MV2DM(args)
    dm.setup("fit")
    model = AutoEncoder(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        prior_model=None,
        args=args
    )
    trainer_model = AutoEncoderTrainer(model, args)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/'),
        callbacks=custom_callbacks(args),
        check_val_every_n_epoch = args.check_val_every_n_epoch,
        detect_anomaly=args.detect_anomaly,
        limit_test_batches=1.0,  # Ensure all test batches are processed
    )

    if args.test_only:
        trainer.test(trainer_model, datamodule=dm)
    else:
        trainer.fit(trainer_model, datamodule=dm)
        trainer.test(trainer_model, datamodule=dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = PCQM4MV2DM.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--node_dim', type=int, default=63)
    parser.add_argument('--edge_dim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--encoder_blocks', type=int, default=8)
    parser.add_argument('--decoder_blocks', type=int, default=2)
    parser.add_argument('--disable_compile', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False, help='Only run the test using the last checkpoint')

    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=5000000)
    parser.add_argument('--save_every_n_epochs', type=int, default=50)
    parser.add_argument('--test_every_n_epochs', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-16)
    parser.add_argument('--scheduler', type=str, default="cosine")
    parser.add_argument('--init_lr', type=float, default=5e-5, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--lr_factor', type=float, default=0.8)
    parser.add_argument('--lr_patience', type=int, default=15)
    parser.add_argument('--t0', type=int, default=100000)
    parser.add_argument('--tmult', type=int, default=2)
    parser.add_argument('--etamin', type=float, default=1e-7)

    parser.add_argument('--checkpoint_path', type=str, default="")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--pair_update', action='store_true', default=False)
    parser.add_argument('--trans_version', type=str, default='v6')
    parser.add_argument('--attn_activation', type=str, default='silu')
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='Mask ratio for the autoencoder')
    parser.add_argument('--dataset', type=str, default='pcqm4mv2', help='Dataset to use')
    parser.add_argument('--denoising_weight', type=float, default=0.1)
    parser.add_argument('--denoising', action='store_true', default=False)
    parser.add_argument('--lr_cosine_length', type=int, default=500000)
    parser.add_argument('--prior_model', action='store_true', default=False)
    parser.add_argument('--pos_mask', action='store_true', default=False)
    parser.add_argument('--delta', type=int, default=1000)
    parser.add_argument('--local_global_distill', action='store_true', default=False)
    parser.add_argument('--local_distill_weight', type=float, default=0.1)
    parser.add_argument('--local_crop_ratio', type=float, default=0.4)
    parser.add_argument('--local_radius', type=float, default=3.0)
    parser.add_argument('--local_min_nodes', type=int, default=6)
    parser.add_argument('--local_max_nodes', type=int, default=0, help='0 means no cap')
    parser.add_argument('--local_proj_dim', type=int, default=256)
    parser.add_argument('--local_num_prototypes', type=int, default=256)
    parser.add_argument('--local_student_temp', type=float, default=0.1)
    parser.add_argument('--local_teacher_temp', type=float, default=0.04)
    parser.add_argument('--local_teacher_momentum', type=float, default=0.996)
    parser.add_argument('--local_center_momentum', type=float, default=0.9)
    parser.add_argument('--local_student_use_mask', action='store_true', default=False)
    parser.add_argument('--use_cls_token', action='store_true', default=False)
    parser.add_argument('--cls_distance', type=float, default=1.0)

    args = parser.parse_args()
    main(args)
