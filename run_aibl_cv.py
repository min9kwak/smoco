# -*- coding: utf-8 -*-
import argparse
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

from configs.aibl import AIBLConfig
from tasks.aibl_cv import AIBLCV

from models.backbone.base import calculate_out_features
from models.backbone.densenet import DenseNetBackbone
from models.backbone.resnet import build_resnet_backbone
from models.head.classifier import LinearClassifier

from utils.gpu import set_gpu


def freeze_bn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            for param in child.parameters():
                param.requires_grad = False
    for n, ch in module.named_children():
        freeze_bn(ch)


def main():
    """Main function for single/distributed linear classification."""

    config = AIBLConfig.parse_arguments()

    pretrained_file = os.path.join(config.pretrained_dir, "ckpt.last.pth.tar")
    setattr(config, 'pretrained_file', pretrained_file)

    pretrained_config = os.path.join(config.pretrained_dir, "configs.json")
    with open(pretrained_config, 'rb') as fb:
        pretrained_config = json.load(fb)

    # inherit pretrained configs
    pretrained_config_names = [
        # data_parser
        # 'data_type', 'root', 'data_info', 'mci_only', 'n_splits', 'n_cv',
        'image_size', 'small_kernel', # 'random_state',
        'intensity', 'crop', 'crop_size', 'rotate', 'flip', 'affine', 'blur', 'blur_std', 'prob',
        # model_parser
        'backbone_type', 'init_features', 'growth_rate', 'block_config', 'bn_size', 'dropout_rate',
        'arch', 'no_max_pool',
        # train
        # 'batch_size',
        # moco / supmoco
        'alphas',
        # others
        'task'
    ]

    for name in pretrained_config_names:
        if name in pretrained_config.keys():
            setattr(config, name, pretrained_config[name])

    config.task = config.task + f'_aibl-cv'

    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    rich.print(config.__dict__)
    config.save()

    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    if config.distributed:
        raise NotImplementedError
    else:
        rich.print(f"Single GPU training.")
        main_worker(0, config=config)  # single machine, single gpu


def main_worker(local_rank: int, config: argparse.Namespace):
    """Single process."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Networks
    if config.backbone_type == 'densenet':
        backbone = DenseNetBackbone(in_channels=1,
                                    init_features=config.init_features,
                                    growth_rate=config.growth_rate,
                                    block_config=config.block_config,
                                    bn_size=config.bn_size,
                                    dropout_rate=config.dropout_rate,
                                    semi=False)
        activation = True
    elif config.backbone_type == 'resnet':
        backbone = build_resnet_backbone(arch=config.arch,
                                         no_max_pool=config.no_max_pool,
                                         in_channels=1,
                                         semi=False)
        activation = False
    else:
        raise NotImplementedError

    if config.small_kernel:
        backbone._fix_first_conv()

    # load pretrained model weights
    backbone.load_weights_from_checkpoint(path=config.pretrained_file, key='backbone')

    if config.freeze_bn:
        freeze_bn(backbone)

    # classifier
    if config.crop_size:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.crop_size)
    else:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.image_size)
    classifier = LinearClassifier(in_channels=out_dim, num_classes=2, activation=activation)
    # classifier = MLPClassifier(in_channels=out_dim, num_classes=2, activation=activation)

    # load pretrained model weights
    classifier.load_weights_from_checkpoint(path=config.pretrained_file, key='classifier')

    # Model (Task)
    model = AIBLCV(backbone=backbone, classifier=classifier, config=config)
    model.prepare(
        local_rank=local_rank,
    )

    # Train & evaluate
    model.run(
        save_every=config.save_every,
    )


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
