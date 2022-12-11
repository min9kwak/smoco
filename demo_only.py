# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import rich
import numpy as np
import pickle
import wandb

import torch
import torch.nn as nn

from configs.finetune import DemoFinetuneConfig
from tasks.demo_classification import DemoClassification

from models.backbone.base import calculate_out_features
from models.backbone.densenet import DenseNetBackbone
from models.backbone.resnet import build_resnet_backbone
from models.backbone.demoencoder import DemoEncoder
from models.head.classifier import LinearDemoClassifier

from datasets.brain import BrainProcessor, Brain
from datasets.transforms import make_transforms

from utils.logging import get_rich_logger
from utils.gpu import set_gpu

# configs
from datasets.samplers import ImbalancedDatasetSampler
from torch.utils.data import DataLoader

local_rank = 0
epochs = 10

# define network
from models.backbone.base import BackboneBase
from utils.initialization import initialize_weights
from collections import OrderedDict


class DemoNetwork(BackboneBase):
    def __init__(self,
                 in_channels: int,
                 hidden: str,
                 num_classes: int) -> None:
        super(DemoNetwork, self).__init__(in_channels)

        self.in_channels = in_channels
        self.hidden = list(int(a) for a in hidden.split(','))
        self.out_features = self.hidden[-1]
        self.num_classes = num_classes

        self.layers = None
        self.layers = self.make_layers(self)
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(self):
        layers = nn.Sequential()

        input_layer = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("input", nn.Linear(self.in_channels, self.hidden[0])),
                    ("bn", nn.BatchNorm1d(num_features=self.hidden[0])),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        layers.add_module("input", input_layer)

        if len(self.hidden) > 1:
            for i in range(len(self.hidden) - 1):
                block = nn.Sequential(
                    OrderedDict(
                        [
                            ("linear", nn.Linear(self.hidden[i], self.hidden[i + 1])),
                            ("bn", nn.BatchNorm1d(self.hidden[i + 1])),
                            ("relu", nn.ReLU(inplace=True)),
                        ]
                    )
                )
                layers.add_module(f"block{i}", block)

        layers.add_module('linear', nn.Linear(self.out_features, self.num_classes))
        return layers

    def forward(self, x: torch.FloatTensor):
        return self.layers(x)


demonet = DemoNetwork(in_channels=7, hidden="4", num_classes=2)
demonet.to(local_rank)
adjusted_result = dict()

for random_state in range(2021, 2041, 2):
    from datasets.brain import BrainProcessor, Brain
    data_processor = BrainProcessor(root='D:/data/ADNI',
                               data_info='labels/data_info.csv',
                               data_type='pet',
                               mci_only=True,
                               random_state=2021)
    datasets = data_processor.process(10, 0)

    train_set = Brain(dataset=datasets['train'], data_type='pet', transform=None)
    test_set = Brain(dataset=datasets['test'], data_type='pet', transform=None)
    class_weight = torch.tensor(data_processor.class_weight, dtype=torch.float).to(local_rank)
    loss_function = nn.CrossEntropyLoss(weight=class_weight)

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

    optimizer = get_optimizer(params=[{'params': demonet.parameters()}], name='adamw', lr=0.0001, weight_decay=0)
    scheduler = get_cosine_scheduler(optimizer, epochs=epochs, warmup_steps=0, cycles=1, min_lr=0.0)

    train_sampler = ImbalancedDatasetSampler(dataset=train_set)
    train_loader = DataLoader(dataset=train_set, batch_size=16, sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=16, drop_last=False)

    history = {'train': [], 'test': [], 'adjusted': []}
    for epoch in range(1, epochs + 1):

        print('epoch: ', epoch)
        # train
        demonet.train()
        steps = len(train_loader)
        result = {'loss': torch.zeros(steps, device=local_rank)}

        y_true, y_pred = [], []
        for i, batch in enumerate(train_loader):
            demo = batch['demo'].float().to(local_rank)
            y = batch['y'].to(local_rank)
            logits = demonet(demo)
            loss = loss_function(logits, y.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            result['loss'][i] = loss.detach()
            y_true.append(y.long())
            y_pred.append(logits)
        out = {k: v.mean().item() for k, v in result.items()}

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)
        clf_result = classification_result(y_true=y_true.cpu().numpy(),
                                           y_pred=y_pred.softmax(1).detach().cpu().numpy(),
                                           adjusted=False)
        for k, v in clf_result.items():
            out[k] = v
        history['train'].append(out)

        # eval
        demonet.eval()
        steps = len(test_loader)
        result = {'loss': torch.zeros(steps, device=local_rank)}

        y_true, y_pred = [], []
        for i, batch in enumerate(test_loader):
            demo = batch['demo'].float().to(local_rank)
            y = batch['y'].to(local_rank)
            logits = demonet(demo)
            loss = loss_function(logits, y.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            result['loss'][i] = loss.detach()
            y_true.append(y.long())
            y_pred.append(logits)
        out = {k: v.mean().item() for k, v in result.items()}

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)
        clf_result = classification_result(y_true=y_true.cpu().numpy(),
                                           y_pred=y_pred.softmax(1).detach().cpu().numpy(),
                                           adjusted=False)
        for k, v in clf_result.items():
            out[k] = v
        history['test'].append(out)
        print(out['auroc'])

    # adjusted
    demonet.eval()
    steps = len(test_loader)
    result = {'loss': torch.zeros(steps, device=local_rank)}

    y_true, y_pred = [], []
    for i, batch in enumerate(test_loader):
        demo = batch['demo'].float().to(local_rank)
        y = batch['y'].to(local_rank)
        logits = demonet(demo)
        loss = loss_function(logits, y.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        result['loss'][i] = loss.detach()
        y_true.append(y.long())
        y_pred.append(logits)
    out = {k: v.mean().item() for k, v in result.items()}

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0).to(torch.float32)
    clf_result = classification_result(y_true=y_true.cpu().numpy(),
                                       y_pred=y_pred.softmax(1).detach().cpu().numpy(),
                                       adjusted=True)
    for k, v in clf_result.items():
        out[k] = v
    history['adjusted'].append(out)
    print('adjusted auroc: ', out['auroc'])

    adjusted_result[random_state] = history

auroc, acc, sens, spec = [], [], [], []
for k, v in adjusted_result.items():
    auroc.append(v['adjusted'][0]['auroc'])
    acc.append(v['adjusted'][0]['acc'])
    sens.append(v['adjusted'][0]['sens'])
    spec.append(v['adjusted'][0]['spec'])

np.mean(auroc)
np.mean(acc)
np.mean(sens)
np.mean(spec)