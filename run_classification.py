# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from configs.classification import ClassificationConfig
from tasks.classification import Classification

from models.backbone.base import calculate_out_features
from models.backbone.densenet import DenseNetBackbone
from models.backbone.resnet import build_resnet_backbone
from models.head.classifier import LinearClassifier

from datasets.mri import MRI, MRIProcessor
from datasets.pet import PET, PETProcessor

from datasets.transforms import make_transforms, compute_statistics

from utils.logging import get_rich_logger
from utils.gpu import set_gpu


def main():
    """Main function for single/distributed linear classification."""

    config = ClassificationConfig.parse_arguments()

    config.task = config.data_type + f'-{config.segment}'

    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    # str -> list arguments
    if config.model_name == 'densenet':
        setattr(config, 'block_config', tuple(int(a) for a in config.block_config.split(',')))

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


def main_worker(local_rank: int, config: object):
    """Single process."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    if config.enable_wandb:
        wandb.init(
            name=f'{config.model_name} : {config.hash}',
            project=f'sttr-{config.task}',
            config=config.__dict__
        )

    # Networks
    if config.model_name == 'densenet':
        backbone = DenseNetBackbone(in_channels=1,
                                    init_features=config.init_features,
                                    growth_rate=config.growth_rate,
                                    block_config=config.block_config,
                                    bn_size=config.bn_size,
                                    dropout_rate=config.dropout_rate,
                                    semi=config.semi)
        activation = True
    elif config.model_name == 'resnet':
        backbone = build_resnet_backbone(arch=config.arch,
                                         no_max_pool=config.no_max_pool,
                                         in_channels=1,
                                         semi=config.semi)
        activation = False
    else:
        raise NotImplementedError

    if config.small_kernel:
        backbone._fix_first_conv()

    out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.image_size)
    classifier = LinearClassifier(in_channels=out_dim, num_classes=2, activation=activation)

    # load data
    if config.data_type == 'mri':
        PROCESSOR = MRIProcessor
        DATA = MRI
    elif config.data_type == 'pet':
        PROCESSOR = PETProcessor
        DATA = PET
    else:
        raise NotImplementedError

    data_processor = PROCESSOR(root=config.root,
                               data_info=config.data_info,
                               mci_only=config.mci_only,
                               segment=config.segment,
                               random_state=config.random_state)

    datasets = data_processor.process(train_size=config.train_size)

    if config.intensity == 'normalize':
        mean_std = compute_statistics(DATA=DATA, normalize_set=datasets['train'])
        min_max = (None, None)
    elif config.intensity == 'minmax':
        mean_std = (None, None)
        if config.data_type == 'pet':
            min_max = (-0.1658, +5.5918)
        elif config.data_type == 'mri':
            min_max = (0.0, 255.0)
    else:
        mean_std = (None, None)
        min_max = (None, None)

    train_transform, test_transform = make_transforms(image_size=config.image_size,
                                                      intensity=config.intensity,
                                                      mean_std=mean_std,
                                                      min_max=min_max,
                                                      crop=config.crop,
                                                      flip=config.flip,
                                                      affine=config.affine,
                                                      blur=config.blur,
                                                      blur_std=config.blur_std,
                                                      prob=config.prob)

    train_set = DATA(dataset=datasets['train'], pin_memory=config.pin_memory, transform=train_transform)
    test_set = DATA(dataset=datasets['test'], pin_memory=config.pin_memory, transform=test_transform)

    # Reconfigure batch-norm layers
    if config.balance:
        class_weight = torch.tensor(data_processor.class_weight, dtype=torch.float).to(local_rank)
        loss_function = nn.CrossEntropyLoss(weight=class_weight)
    else:
        loss_function = nn.CrossEntropyLoss()

    # Model (Task)
    model = Classification(backbone=backbone, classifier=classifier)
    model.prepare(
        checkpoint_dir=config.checkpoint_dir,
        loss_function=loss_function,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        cosine_warmup=config.cosine_warmup,
        cosine_cycles=config.cosine_cycles,
        cosine_min_lr=config.cosine_min_lr,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        distributed=config.distributed,
        local_rank=local_rank,
        mixed_precision=config.mixed_precision,
        enable_wandb=config.enable_wandb
    )

    # Train & evaluate
    start = time.time()
    model.run(
        train_set=train_set,
        test_set=test_set,
        save_every=config.save_every,
        logger=logger
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
