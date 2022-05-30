# -*- coding: utf-8 -*-

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import classification_result

from torch.optim.swa_utils import AveragedModel, SWALR

from datasets.samplers import ImbalancedDatasetSampler

from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from utils.logging import get_rich_pbar

import wandb


class SWA(object):
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 l_loss_function: nn.Module,
                 u_loss_function: nn.Module
                 ):
        super(SWA, self).__init__()

        # Initialize networks
        self.backbone = backbone
        self.classifier = classifier

        self.l_loss_function = l_loss_function
        self.u_loss_function = u_loss_function

        self.scaler = None
        self.optimizer = None
        self.scheduler = None

        self.swa_backbone = None
        self.swa_classifier = None
        self.swa_scheduler = None
        self.swa_start = None

        self.prepared = False

    def prepare(self,
                checkpoint_dir: str,
                optimizer: str,
                learning_rate: float = 0.01,
                weight_decay: float = 1e-4,
                cosine_warmup: int = 0,
                cosine_cycles: int = 1,
                cosine_min_lr: float = 5e-3,
                epochs: int = 200,
                swa_learning_rate: float = 1e-4,
                swa_start: int = 0,
                batch_size: int = 16,
                mu: int = 3,
                alpha: float = 1.0,
                ramp_up: int = 5,
                num_workers: int = 4,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                enable_wandb: bool = True,
                resume: str = None):

        # Set attributes
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.mu = mu
        self.alpha = alpha
        self.ramp_up = ramp_up
        self.num_workers = num_workers
        self.distributed = distributed
        self.local_rank = local_rank
        self.mixed_precision = mixed_precision
        self.enable_wandb = enable_wandb
        self.resume = resume

        # optimizer and scheduler
        self.optimizer = get_optimizer(
            params=[
                {'params': self.backbone.parameters()},
                {'params': self.classifier.parameters()},
            ],
            name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            epochs=self.epochs,
            warmup_steps=cosine_warmup,
            cycles=cosine_cycles,
            min_lr=cosine_min_lr,
        )

        # SWA settings
        self.swa_backbone = AveragedModel(self.backbone)
        self.swa_classifier = AveragedModel(self.classifier)
        self.swa_scheduler = SWALR(optimizer=self.optimizer, swa_lr=swa_learning_rate)
        self.swa_start = swa_start

        # Resuming from previous checkpoint (optional)
        if resume is not None:
            if not os.path.exists(resume):
                raise FileNotFoundError
            self.load_model_from_checkpoint(resume)

        # Distributed training (optional, disabled by default.)
        if distributed:
            raise NotImplementedError

        # No DDP wrapping for key encoder, as it does not have gradients
        self.backbone.to(local_rank)
        self.classifier.to(local_rank)
        self.swa_backbone.to(local_rank)
        self.swa_classifier.to(local_rank)

        # Mixed precision training (optional, enabled by default.)
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Ready to train!
        self.prepared = True

    def run(self,
            l_train_set: torch.utils.data.Dataset,
            u_train_set: torch.utils.data.Dataset,
            test_set: torch.utils.data.Dataset,
            save_every: int = 100,
            **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (for self-supervised pre-training)
        l_train_sampler = ImbalancedDatasetSampler(dataset=l_train_set)
        l_train_loader = DataLoader(dataset=l_train_set, batch_size=self.batch_size,
                                    sampler=l_train_sampler, num_workers=self.num_workers, drop_last=True,
                                    pin_memory=True)
        u_train_loader = DataLoader(dataset=u_train_set, batch_size=self.batch_size * self.mu, shuffle=True,
                                    num_workers=self.num_workers, drop_last=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size * 2, shuffle=False,
                                 num_workers=self.num_workers, drop_last=False, pin_memory=False)

        assert len(u_train_loader) >= len(l_train_loader)

        # Logging
        logger = kwargs.get('logger', None)

        if self.enable_wandb:
            wandb.watch([self.backbone, self.classifier], log='all', log_freq=len(u_train_loader))

        for epoch in range(1, self.epochs + 1):

            self.epoch = epoch

            # Train
            train_history = self.train(l_train_loader, u_train_loader)

            # after training an epoch, update scheduler either swa or basic
            if self.scheduler is not None:
                if epoch > self.swa_start:
                    self.swa_backbone.update_parameters(self.backbone)
                    self.swa_classifier.update_parameters(self.classifier)
                    self.swa_scheduler.step()
                    update_bn(l_train_loader, self.swa_backbone, device=self.local_rank)
                else:
                    self.scheduler.step()

            test_history = self.evaluate(test_loader)

            # Logging
            log = " | ".join([f"train/{k} : {v:.4f}" for k, v in train_history.items()])
            for metric, score in test_history.items():
                log += f" | test/{metric}: {score:.4f}"

            if logger is not None:
                logger.info(f"Epoch [{epoch:>4}/{self.epochs:>4}] - " + log)

            # TensorBoard
            epoch_history = collections.defaultdict(dict)
            for k, v1 in train_history.items():
                epoch_history[k]['train'] = v1
                try:
                    v2 = test_history[k]
                    epoch_history[k]['test'] = v2
                except KeyError:
                    continue

            # TODO:change in-out of epoch_history

            if self.enable_wandb:
                wandb.log({'epoch': epoch}, commit=False)
                if self.scheduler is not None:
                    wandb.log({'lr': self.scheduler.get_last_lr()[0]}, commit=False)
                wandb.log(epoch_history)

            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.checkpoint_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch, history=epoch_history)

        # save last
        if self.local_rank == 0:
            ckpt = os.path.join(self.checkpoint_dir, f"ckpt.last.pth.tar")
            self.save_checkpoint(ckpt, epoch=epoch, history=epoch_history)

    def train(self, l_train_loader: torch.utils.data.DataLoader, u_train_loader: torch.utils.data.DataLoader):
        """Training SWA based on pi-model."""

        steps = len(u_train_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'l_loss': torch.zeros(steps, device=self.local_rank),
            'u_loss': torch.zeros(steps, device=self.local_rank),
        }

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Waiting... ", total=steps)

            l_train_iter = iter(l_train_loader)
            y_true, y_pred = [], []
            for i, u_batch in enumerate(u_train_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    try:
                        l_batch = next(l_train_iter)
                    except StopIteration:
                        l_train_iter = iter(l_train_loader)
                        l_batch = next(l_train_iter)

                    l_x1 = l_batch['x1'].to(self.local_rank)
                    u_x1 = u_batch['x1'].to(self.local_rank)
                    l_x2 = l_batch['x2'].to(self.local_rank)
                    u_x2 = u_batch['x2'].to(self.local_rank)

                    l_y = l_batch['y'].to(self.local_rank).long()
                    u_y = u_batch['y'].to(self.local_rank).long()

                    # ssl loss
                    y = torch.cat([l_y, u_y], dim=0)
                    mask = (y == -1).float()

                    y_pred_1 = self.classifier(self.backbone(torch.cat([l_x1, u_x1], dim=0)))

                    # calculate loss
                    l_loss = self.l_loss_function(y_pred_1, y).mean()
                    u_loss = self.u_loss_function(torch.cat([l_x2, u_x2], dim=0), y_pred_1,
                                                  self.backbone, self.classifier, mask)

                    # calculate coefficient ramp_up
                    if self.epoch < self.ramp_up:
                        numerator = (self.epoch - 1) * steps + i
                        denominator = (self.ramp_up - 1) * steps
                        mul = numerator / denominator
                    else:
                        mul = 1
                    loss = l_loss + u_loss * self.alpha * mul

                    self.backprop(loss)

                    # store for scoring
                    y_true.append(l_y)
                    y_pred.append(y_pred_1[y != -1])

                result['loss'][i] = loss.detach()
                result['l_loss'][i] = l_loss.detach()
                result['u_loss'][i] = u_loss.detach()

                if self.local_rank == 0:
                    desc = f"[bold green] [{i + 1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        # scoring
        out = {k: v.mean().item() for k, v in result.items()}

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)

        clf_result = classification_result(y_true=y_true, y_pred=y_pred)
        for k, v in clf_result.items():
            out[k] = v

        return out

    @torch.no_grad()
    def evaluate(self, loader):

        self._set_learning_phase(False)

        if self.epoch < self.swa_start:
            eval_backbone = self.backbone
            eval_classifier = self.classifier
        else:
            eval_backbone = self.swa_backbone
            eval_classifier = self.swa_classifier

        y, y_pred = [], []
        for batch in loader:
            x = batch['x1'].to(self.local_rank)
            y.append(batch['y'].to(self.local_rank).long())
            y_pred.append(eval_classifier(eval_backbone(x)))
        y = torch.cat(y, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)

        # calculate accuracy and f1 scores after softmax
        out = dict()
        clf_result = classification_result(y_true=y, y_pred=y_pred)
        for k, v in clf_result.items():
            out[k] = v

        return out

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
            self.backbone.train()
            self.classifier.train()
            self.swa_backbone.train()
            self.swa_classifier.train()
        else:
            self.backbone.eval()
            self.classifier.eval()
            self.swa_backbone.eval()
            self.swa_classifier.eval()

    def save_checkpoint(self, path: str, **kwargs):
        """Save model to a `.tar' checkpoint file."""

        ckpt = {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'swa_backbone': self.swa_backbone.state_dict(),
            'swa_classifier': self.swa_classifier.state_dict(),
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
        self.backbone.load_state_dict(ckpt['backbone'])
        self.classifier.load_state_dict(ckpt['classifier'])
        self.swa_backbone.load_state_dict(ckpt['swa_backbone'])
        self.swa_classifier.load_state_dict(ckpt['swa_classifier'])
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


@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        input = input['x1']
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
