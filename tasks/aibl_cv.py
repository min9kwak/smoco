import argparse
import os
import collections
import time

import torch
import torch.nn as nn

from datasets.aibl import AIBLProcessor, AIBLDataset
from datasets.transforms import make_transforms
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from utils.metrics import classification_result

import wandb
from utils.logging import make_epoch_description, get_rich_pbar
from datasets.samplers import ImbalancedDatasetSampler
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler

from utils.logging import get_rich_logger


class AIBLCV(object):
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 config: argparse.Namespace
                 ):

        # network
        self.backbone = backbone
        self.classifier = classifier

        # optimizer
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None

        self.config = config

        # prepared
        self.prepared = False

    def prepare(self,
                local_rank: int = 0,
                **kwargs):  # pylint: disable=unused-argument

        # Set attributes
        self.checkpoint_dir = self.config.checkpoint_dir
        self.epochs = self.config.epochs
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.local_rank = local_rank
        self.mixed_precision = self.config.mixed_precision
        self.enable_wandb = self.config.enable_wandb

        # Distributed training (optional)
        if self.config.distributed:
            raise NotImplementedError
        else:
            self.backbone.to(self.local_rank)
            self.classifier.to(self.local_rank)

        # Optimization
        self.optimizer = get_optimizer(
            params=[
                {'params': self.backbone.parameters()},
                {'params': self.classifier.parameters()},
            ],
            name=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            epochs=self.epochs,
            warmup_steps=self.config.cosine_warmup,
            cycles=self.config.cosine_cycles,
            min_lr=self.config.cosine_min_lr,
            )
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        # Ready to train!
        self.prepared = True

    def run(self,
            save_every: int = 10,
            **kwargs):

        epochs = self.epochs

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        y_true_final, y_pred_final = [], []

        for n_cv in range(self.config.n_splits):

            setattr(self.config, 'n_cv', n_cv)

            start = time.time()

            if self.local_rank == 0:
                logfile = os.path.join(self.config.checkpoint_dir, f'main_{n_cv}.log')
                logger = get_rich_logger(logfile=logfile)
                if self.config.enable_wandb:
                    wandb.init(
                        name=f'{self.config.backbone_type} : {self.config.hash}',
                        project=f'sttr-{self.config.task}',
                        config=self.config.__dict__
                    )
            else:
                logger = None

            train_set, test_set = self.set_dataset(n_cv=n_cv)

            if train_set is not None:
                train_sampler = ImbalancedDatasetSampler(dataset=train_set)
                train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size,
                                          sampler=train_sampler, num_workers=self.num_workers,
                                          drop_last=True)
            else:
                train_loader = None
            test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                                     drop_last=False)

            # Supervised training
            best_eval_loss = float('inf')
            best_epoch = 0

            if (self.enable_wandb) and (train_set is not None):
                wandb.watch([self.backbone, self.classifier], log='all', log_freq=len(train_loader))

            if self.config.train_mode == 'train':
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
                            ckpt = os.path.join(self.checkpoint_dir, f"ckpt_{n_cv}.best.pth.tar")
                            self.save_checkpoint(ckpt, epoch=epoch)

                    # Save intermediate model checkpoints
                    if (epoch % save_every == 0) & (self.local_rank == 0):
                        ckpt = os.path.join(self.checkpoint_dir, f"ckpt_{n_cv}.{epoch}.pth.tar")
                        self.save_checkpoint(ckpt, epoch=epoch)

                    # Update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()

                # Save final model checkpoint
                ckpt = os.path.join(self.checkpoint_dir, f"ckpt_{n_cv}.last.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch)

            else:
                epoch = 1
                test_history = self.evaluate(test_loader)
                epoch_history = collections.defaultdict(dict)
                for k, v1 in test_history.items():
                    epoch_history[k]['test'] = v1

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

            # adjusted evaluation
            test_history, y_true, y_pred = self.evaluate(test_loader, adjusted=True, return_values=True)
            y_true_final.append(y_true)
            y_pred_final.append(y_pred)

            epoch_history = collections.defaultdict(dict)
            for k, v1 in test_history.items():
                epoch_history[k]['adjusted'] = v1

            if self.enable_wandb:
                log_history = collections.defaultdict(dict)
                for metric_name, scores in epoch_history.items():
                    for mode, value in scores.items():
                        log_history[f'{mode}/{metric_name}'] = value
                wandb.log(log_history)

            elapsed_sec = time.time() - start

            if logger is not None:
                elapsed_mins = elapsed_sec / 60
                logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
                logger.handlers.clear()

        # TODO: concatenate (final)
        y_true_final = torch.cat(y_true_final, dim=0)
        y_pred_final = torch.cat(y_pred_final, dim=0)

        clf_result = classification_result(y_true=y_true_final.cpu().numpy(),
                                           y_pred=y_pred_final.softmax(1).detach().cpu().numpy(),
                                           adjusted=False)
        clf_result_adj = classification_result(y_true=y_true_final.cpu().numpy(),
                                               y_pred=y_pred_final.softmax(1).detach().cpu().numpy(),
                                               adjusted=True)

        final_history = collections.defaultdict(dict)
        for k, v in clf_result.items():
            final_history[f'final/plain/{k}'] = v
        for k, v in clf_result_adj.items():
            final_history[f'final/adjusted/{k}'] = v
        wandb.log(final_history)

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
                    x = batch['x'].float().to(self.local_rank)
                    y = batch['y'].to(self.local_rank)
                    logits = self.classifier(self.backbone(x))
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

                y_true.append(y.long())
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
    def evaluate(self, data_loader, adjusted=False, return_values=False):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
        }

        y_true, y_pred = [], []
        for i, batch in enumerate(data_loader):

            x = batch['x'].float().to(self.local_rank)
            y = batch['y'].to(self.local_rank)
            logits = self.classifier(self.backbone(x))
            loss = self.loss_function(logits, y.long())

            result['loss'][i] = loss.detach()

            y_true.append(y.long())
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

        if return_values:
            return out, y_true, y_pred
        else:
            return out

    def set_dataset(self, n_cv):
        # load finetune data
        data_processor = AIBLProcessor(root=self.config.root,
                                       data_info=self.config.data_info,
                                       time_window=self.config.time_window,
                                       random_state=self.config.random_state)
        test_only = True if self.config.train_mode == 'test' else False
        datasets = data_processor.process(n_splits=self.config.n_splits, n_cv=n_cv, test_only=test_only)

        # intensity normalization
        assert self.config.intensity in [None, 'scale', 'minmax', 'normalize']
        mean_std, min_max = (None, None), (None, None)

        train_transform, test_transform = make_transforms(image_size=self.config.image_size,
                                                          intensity=self.config.intensity,
                                                          min_max=min_max,
                                                          crop_size=self.config.crop_size,
                                                          rotate=self.config.rotate,
                                                          flip=self.config.flip,
                                                          affine=self.config.affine,
                                                          blur_std=self.config.blur_std,
                                                          prob=self.config.prob)

        finetune_transform = train_transform if self.config.finetune_trans == 'train' else test_transform
        if not test_only:
            train_set = AIBLDataset(dataset=datasets['train'], transform=finetune_transform)
        else:
            train_set = None
        test_set = AIBLDataset(dataset=datasets['test'], transform=test_transform)

        # Reconfigure batch-norm layers
        if (self.config.balance) and (not test_only):
            class_weight = torch.tensor(data_processor.class_weight, dtype=torch.float).to(local_rank)
            loss_function = nn.CrossEntropyLoss(weight=class_weight)
        else:
            loss_function = nn.CrossEntropyLoss()
        self.loss_function = loss_function

        return train_set, test_set

    def _set_learning_phase(self, train=False):
        if train:
            self.backbone.train()
            self.classifier.train()
        else:
            self.backbone.eval()
            self.classifier.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.classifier.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    @staticmethod
    def freeze_params(net: nn.Module):
        for p in net.parameters():
            p.requires_grad = False
