import os
import collections
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from utils.metrics import classification_result

import wandb
from utils.logging import make_epoch_description, get_rich_pbar
from datasets.samplers import ImbalancedDatasetSampler
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler


class SliceClassification(object):
    def __init__(self,
                 network: nn.Module,
                 ):

        # network
        self.network = network

        # optimizer
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None

        # prepared
        self.prepared = False

    def prepare(self,
                checkpoint_dir: str,
                loss_function: nn.Module,
                optimizer: str = 'adamw',
                learning_rate: float = 0.01,
                weight_decay: float = 0.001,
                cosine_warmup: int = 0,
                cosine_cycles: int = 1,
                cosine_min_lr: float = 0.0,
                epochs: int = 100,
                batch_size: int = 4,
                num_workers: int = 4,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                enable_wandb: bool = True,
                **kwargs):  # pylint: disable=unused-argument

        # Set attributes
        self.checkpoint_dir = checkpoint_dir
        self.loss_function = loss_function
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.local_rank = local_rank
        self.mixed_precision = mixed_precision
        self.enable_wandb = enable_wandb
        self.config = kwargs.get('config', None)

        if self.config.train_slices == 'random':
            self.test_num_slices = 3
        elif self.config.train_slices == 'fixed':
            self.test_num_slices = 3
        elif self.config.train_slices in ['sagittal', 'coronal', 'axial']:
            self.test_num_slices = 1
        else:
            raise ValueError

        # Distributed training (optional)
        if distributed:
            raise NotImplementedError
        else:
            self.network.to(self.local_rank)

        # Optimization
        self.optimizer = get_optimizer(
            params=[
                {'params': self.network.parameters()},
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
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Ready to train!
        self.prepared = True

    def run(self,
            train_set: torch.utils.data.Dataset,
            test_set: torch.utils.data.Dataset,
            save_every: int = 10,
            **kwargs):

        # TODO: implement eval_set
        epochs = self.epochs

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (train, val, test)
        train_sampler = ImbalancedDatasetSampler(dataset=train_set)
        train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size,
                                  sampler=train_sampler, num_workers=self.num_workers,
                                  drop_last=True)

        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                                 drop_last=False)

        # Logging
        logger = kwargs.get('logger', None)

        # Supervised training
        best_eval_loss = float('inf')
        best_epoch = 0

        if self.enable_wandb:
            wandb.watch([self.network], log='all', log_freq=len(train_loader))

        for epoch in range(1, epochs + 1):

            # Train & evaluate
            train_history = self.train(train_loader)
            test_history = self.evaluate(test_loader)

            epoch_history = collections.defaultdict(dict)
            for k, v1 in train_history.items():
                epoch_history[k]['train'] = v1
                try:
                    v2 = test_history[k]
                    epoch_history[k]['test'] = v2
                except KeyError:
                    continue

            # Write logs
            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

            if self.enable_wandb:
                wandb.log({'epoch': epoch}, commit=False)
                if self.scheduler is not None:
                    wandb.log({'lr': self.scheduler.get_last_lr()[0]}, commit=False)
                else:
                    wandb.log({'lr': self.optimizer.param_groups[0]['lr']}, commit=False)

                log_history = collections.defaultdict(dict)
                for metric_name, scores in epoch_history.items():
                    for mode, value in scores.items():
                        log_history[f'{mode}/{metric_name}'] = value
                wandb.log(log_history)

            # Save best model checkpoint
            eval_loss = test_history['loss']
            if eval_loss <= best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.checkpoint_dir, f"ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

            # Save intermediate model checkpoints
            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.checkpoint_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model checkpoint
        ckpt = os.path.join(self.checkpoint_dir, f"ckpt.last.pth.tar")
        self.save_checkpoint(ckpt, epoch=epoch)

        # adjusted evaluation
        test_history = self.evaluate(test_loader, adjusted=True)
        epoch_history = collections.defaultdict(dict)
        for k, v1 in test_history.items():
            epoch_history[k]['adjusted'] = v1

        if self.enable_wandb:
            log_history = collections.defaultdict(dict)
            for metric_name, scores in epoch_history.items():
                for mode, value in scores.items():
                    log_history[f'{mode}/{metric_name}'] = value
            wandb.log(log_history)

    def train(self, data_loader):
        """Training defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank)
        }

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    x = torch.concat(batch['x']).float().to(self.local_rank)
                    y = batch['y'].repeat(self.config.num_slices).to(self.local_rank)

                    logits = self.network(x)
                    loss = self.loss_function(logits, y.long())
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                result['loss'][i] = loss.detach()
                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # save true / pred (actual number of observations)
                y_true.append(y.chunk(self.config.num_slices)[0].long())
                num_classes = logits.shape[-1]
                logits = logits.softmax(1)
                logits = logits.reshape(self.config.num_slices, -1, num_classes).mean(0)
                y_pred.append(logits)

        out = {k: v.mean().item() for k, v in result.items()}

        # enforce to float32: accuracy and macro f1 score
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)

        clf_result = classification_result(y_true=y_true.cpu().numpy(),
                                           y_pred=y_pred.softmax(1).detach().cpu().numpy(),
                                           adjusted=False)
        for k, v in clf_result.items():
            out[k] = v

        return out

    @torch.no_grad()
    def evaluate(self, data_loader, adjusted=False):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
        }

        y_true, y_pred = [], []
        for i, batch in enumerate(data_loader):

            x = torch.concat(batch['x']).float().to(self.local_rank)
            y = batch['y'].repeat(self.test_num_slices).to(self.local_rank)

            logits = self.network(x)
            loss = self.loss_function(logits, y.long())

            result['loss'][i] = loss.detach()

            # save true / pred (actual number of observations)
            y_true.append(y.chunk(self.test_num_slices)[0].long())
            num_classes = logits.shape[-1]
            logits = logits.softmax(1)
            logits = logits.reshape(self.test_num_slices, -1, num_classes).mean(0)
            y_pred.append(logits)

        out = {k: v.mean().item() for k, v in result.items()}

        # accuracy and macro f1 score
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)

        clf_result = classification_result(y_true=y_true.cpu().numpy(),
                                           y_pred=y_pred.softmax(1).detach().cpu().numpy(),
                                           adjusted=adjusted)
        for k, v in clf_result.items():
            out[k] = v

        return out

    def _set_learning_phase(self, train=False):
        if train:
            self.network.train()
        else:
            self.network.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.network.load_state_dict(ckpt['network'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    @staticmethod
    def freeze_params(net: nn.Module):
        for p in net.parameters():
            p.requires_grad = False
