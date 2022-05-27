# -*- coding: utf-8 -*-

import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from utils.distributed import ForMoCo
from utils.distributed import concat_all_gather
from utils.metrics import TopKAccuracy
from utils.knn import KNNEvaluator
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from utils.logging import get_rich_pbar, make_epoch_description

import wandb


class MoCoLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature

    def forward(self,
                queries: torch.FloatTensor,
                keys: torch.FloatTensor,
                queue: torch.FloatTensor):

        # Calculate logits
        pos_logits = torch.einsum('nc,nc->n', [queries, keys]).view(-1, 1)
        neg_logits = torch.einsum('nc,ck->nk', [queries, queue.clone().detach()])
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (B, 1+K)
        logits.div_(self.temperature)

        # Create labels
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Compute loss
        loss = nn.functional.cross_entropy(logits, labels)

        return loss, logits, labels


class MemoryQueue(nn.Module):
    def __init__(self, size: tuple, device: int = 0):
        super(MemoryQueue, self).__init__()

        if len(size) != 2:
            raise ValueError(f"Invalid size for memory: {size}. Only supports 2D.")
        self.size = size
        self.device = device

        with torch.no_grad():
            self.buffer = torch.randn(*self.size, device=self.device)  # (f, K)
            self.buffer = F.normalize(self.buffer, dim=0)  # l2 normalize
            self.ptr = torch.zeros(1, dtype=torch.long, device=self.device)
            self.labels = torch.zeros(self.size[1], dtype=torch.long, device=self.device)  # (K, )

        self.num_updates = 0
        self.is_reliable = False

    @property
    def num_negatives(self):
        return self.buffer.size(1)

    @torch.no_grad()
    def update(self, keys: torch.FloatTensor, labels: torch.LongTensor = None):
        """
        Update memory queue shared along processes.
        Arguments:
            keys: torch.FloatTensor of shape (B, f)
        """
        if labels is not None:
            assert len(keys) == len(labels), print(keys.shape, labels.shape)

        # Gather along multiple processes.
        keys = concat_all_gather(keys)  # (B, f) -> (world_size * B, f)
        incoming, _ = keys.size()
        if self.num_negatives % incoming != 0:
            raise ValueError("Use exponentials of 2 for number of negatives.")

        # Update queue (keys, and optionally labels if provided)
        ptr = int(self.ptr)
        self.buffer[:, ptr: ptr + incoming] = keys.T
        if labels is not None:
            labels = concat_all_gather(labels)  # (B, ) -> (world_size * B, )
            self.labels[ptr: ptr + incoming] = labels

        # Check if the current queue is reliable
        if not self.is_reliable:
            self.is_reliable = (ptr + incoming) >= self.num_negatives

        # Update pointer
        ptr = (ptr + incoming) % self.num_negatives
        self.ptr[0] = ptr
        self.num_updates += 1


class MoCo(object):
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module,
                 queue: MemoryQueue,
                 loss_function: nn.Module,
                 ):
        super(MoCo, self).__init__()

        # Initialize networks
        self.queue = queue
        self.net_q = nn.Sequential()
        self.net_q.add_module('backbone', backbone)
        self.net_q.add_module('head', head)
        self.net_k = copy.deepcopy(self.net_q)
        self._freeze_key_net_params()

        self.loss_function = loss_function

        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None

        self.prepared = False

    def prepare(self,
                checkpoint_dir: str,
                optimizer: str,
                learning_rate: float = 0.01,
                weight_decay: float = 1e-4,
                cosine_warmup: int = 0,
                cosine_cycles: int = 1,
                cosine_min_lr: float = 0.0,
                epochs: int = 1000,
                batch_size: int = 16,
                num_workers: int = 4,
                key_momentum: float = 0.999,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = False,
                enable_wandb: bool = True,
                resume: str = None):
        """Prepare MoCo pre-training."""

        # Set attributes
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.key_momentum = key_momentum
        self.distributed = distributed
        self.local_rank = local_rank
        self.mixed_precision = mixed_precision
        self.enable_wandb = enable_wandb
        self.resume = resume

        self.optimizer = get_optimizer(
            params=self.net_q.parameters(),
            name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduling; if cosine_warmup < 0: scheduler = None.
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            epochs=self.epochs,
            warmup_steps=cosine_warmup,
            cycles=cosine_cycles,
            min_lr=cosine_min_lr,
        )

        # Resuming from previous checkpoint (optional)
        if resume is not None:
            if not os.path.exists(resume):
                raise FileNotFoundError
            self.load_model_from_checkpoint(resume)

        # Distributed training (optional, disabled by default.)
        if distributed:
            self.net_q = DistributedDataParallel(
                module=self.net_q.to(local_rank),
                device_ids=[local_rank]
            )
        else:
            self.net_q.to(local_rank)

        # No DDP wrapping for key encoder, as it does not have gradients
        self.net_k.to(local_rank)

        # Mixed precision training (optional, enabled by default.)
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Ready to train!
        self.prepared = True

    def run(self,
            dataset: torch.utils.data.Dataset,
            memory_set: torch.utils.data.Dataset = None,
            query_set: torch.utils.data.Dataset = None,
            save_every: int = 50,
            **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (for self-supervised pre-training)
        sampler = DistributedSampler(dataset) if self.distributed else None
        shuffle = not self.distributed
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

        # DataLoader (for supervised evaluation): memory_loader -> validation set & query_loader -> test set
        memory_loader = DataLoader(memory_set, batch_size=self.batch_size, num_workers=self.num_workers)
        query_loader = DataLoader(query_set, batch_size=self.batch_size, num_workers=self.num_workers)

        # Logging
        logger = kwargs.get('logger', None)

        if self.enable_wandb:
            wandb.watch([self.net_k], log='all', log_freq=len(train_loader))

        for epoch in range(1, self.epochs + 1):

            if self.distributed and (sampler is not None):
                sampler.set_epoch(epoch)

            # Train
            history = self.train(train_loader)
            log = " | ".join([f"{k} : {v:.4f}" for k, v in history.items()])

            # Evaluate
            if self.local_rank == 0:
                knn_k = kwargs.get('knn_k', [1, 5, 15])
                knn = KNNEvaluator(knn_k, num_classes=2)
                knn_scores = knn.evaluate(self.net_q,
                                          memory_loader=memory_loader,
                                          query_loader=query_loader)
                for k, score in knn_scores.items():
                    log += f" | {k}: {score * 100:.2f}%"
            else:
                knn_scores = None

            # Logging
            if logger is not None:
                logger.info(f"Epoch [{epoch:>4}/{self.epochs:>4}] - " + log)

            # TensorBoard
            if self.enable_wandb:
                wandb.log({'epoch': epoch}, commit=False)
                wandb.log(history, commit=False)
                if self.scheduler is not None:
                    wandb.log({'lr': self.scheduler.get_last_lr()[0]}, commit=False)
                if knn_scores is not None:
                    wandb.log(knn_scores)

            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.checkpoint_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch, history=history)

            if self.scheduler is not None:
                self.scheduler.step()

        # save last
        if self.local_rank == 0:
            ckpt = os.path.join(self.checkpoint_dir, f"ckpt.last.pth.tar")
            self.save_checkpoint(ckpt, epoch=epoch, history=history)

    def train(self, data_loader: torch.utils.data.DataLoader):
        """MoCo training."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'rank@1': torch.zeros(steps, device=self.local_rank),
        }

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Waiting... ", total=steps)

            for i, batch in enumerate(data_loader):

                loss, logits, labels = self.train_step(batch)
                result['loss'][i] = loss.detach()
                result['rank@1'][i] = TopKAccuracy(k=1)(logits, labels)

                if self.local_rank == 0:
                    desc = f"[bold green] [{i + 1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        return {k: v.mean().item() for k, v in result.items()}

    def train_step(self, batch: dict):
        """A single forward & backward pass."""

        with torch.cuda.amp.autocast(self.mixed_precision):
            # Get data (two views)
            x_q = batch['x1'].to(self.local_rank)
            x_k = batch['x2'].to(self.local_rank)

            # Compute query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)

            with torch.no_grad():
                # Update momentum encoder
                self._momentum_update_key_net()

                # Shuffle across nodes (gpus)
                x_k, idx_unshuffle = ForMoCo.batch_shuffle_ddp(x_k)

                # Compute key features; (B, f)
                z_k = F.normalize(self.net_k(x_k), dim=1)

                # Restore original keys (which were distributed across nodes)
                z_k = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle)

            # Compute loss
            loss, logits, labels = self.loss_function(z_q, z_k, self.queue.buffer)

            # Backpropagate & update
            self.backprop(loss)

            # Update memory queue
            self.queue.update(keys=z_k)

        return loss, logits, labels

    def backprop(self, loss: torch.FloatTensor):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _set_learning_phase(self, train: bool = False):
        if train:
            self.net_q.train()
            self.net_k.train()
        else:
            self.net_q.eval()
            self.net_k.eval()

    @torch.no_grad()
    def _freeze_key_net_params(self):
        """Disable gradient calculation of key network."""
        for p in self.net_k.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_net(self):
        for p_q, p_k in zip(self.net_q.parameters(), self.net_k.parameters()):
            p_k.data = p_k.data * self.key_momentum + p_q.data * (1. - self.key_momentum)

    def save_checkpoint(self, path: str, **kwargs):
        """Save model to a `.tar' checkpoint file."""

        if self.distributed:
            backbone = self.net_q.module.backbone
            head = self.net_q.module.head
        else:
            backbone = self.net_q.backbone
            head = self.net_q.head

        ckpt = {
            'backbone': backbone.state_dict(),
            'head': head.state_dict(),
            'net_k': self.net_k.state_dict(),
            'queue': self.queue.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        """
        Loading model from a checkpoint.
        If resuming training, ensure that all modules have been properly initialized.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.net_q.backbone.load_state_dict(ckpt['backbone'])
        self.net_q.head.load_state_dict(ckpt['head'])
        self.net_k.load_state_dict(ckpt['net_k'])
        self.queue.load_state_dict(ckpt['queue'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.move_optimizer_states(self.optimizer, self.local_rank)

    @staticmethod
    def move_optimizer_states(optimizer: torch.optim.Optimizer, device: int = 0):
        for state in optimizer.state.values():  # dict; state of parameters
            for k, v in state.items():          # iterate over paramteters (k=name, v=tensor)
                if torch.is_tensor(v):          # If a tensor,
                    state[k] = v.to(device)     # configure appropriate device

    @staticmethod
    def freeze_params(net: nn.Module):
        for p in net.parameters():
            p.requires_grad = False
