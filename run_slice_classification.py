# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import pickle
import wandb

import torch
import torch.nn as nn

from configs.slice.classification import SliceClassificationConfig
from tasks.slice.classification import SliceClassification

from models.slice.resnet import resnet50, resnet18
from models.slice.head import LinearClassifier

from datasets.brain import BrainProcessor, Brain
from datasets.slice.transforms import make_transforms

from utils.logging import get_rich_logger
from utils.gpu import set_gpu


def main():
    """Main function for single/distributed linear classification."""

    config = SliceClassificationConfig.parse_arguments()

    config.task = config.data_type + '-SliceClassification'

    if (config.train_slices == 'fixed') and (config.num_slices != 3):
        config.num_slices = 3
        rich.print(f'Number of slices are modified to 3 ({config.train_slices})')
    if (config.train_slices in ['sagittal', 'coronal', 'axial']) and (config.num_slices != 1):
        config.num_slices = 1
        rich.print(f'Number of slices are modified to 1 ({config.train_slices})')

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
    if config.backbone_type == 'resnet50':
        # TODO: initialization
        network = resnet50(num_classes=2)
    elif config.backbone_type == 'resnet18':
        network = resnet18(num_classes=2)
    else:
        raise ValueError
    network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if config.small_kernel:
        conv1 = network.conv1
        network.conv1 = nn.Conv2d(conv1.in_channels, conv1.out_channels,
                                  kernel_size=3, stride=1, padding=1, bias=False)

    # load data
    data_processor = BrainProcessor(root=config.root,
                                    data_info=config.data_info,
                                    data_type=config.data_type,
                                    mci_only=config.mci_only,
                                    random_state=config.random_state)
    datasets = data_processor.process(n_splits=config.n_splits, n_cv=config.n_cv)

    train_transform, test_transform = make_transforms(image_size=config.image_size,
                                                      intensity=config.intensity,
                                                      crop_size=config.crop_size,
                                                      rotate=config.rotate,
                                                      flip=config.flip,
                                                      affine=config.affine,
                                                      blur_std=config.blur_std,
                                                      train_slices=config.train_slices,
                                                      num_slices=config.num_slices,
                                                      slice_range=config.slice_range,
                                                      prob=config.prob)

    train_set = Brain(dataset=datasets['train'], data_type=config.data_type, transform=train_transform)
    test_set = Brain(dataset=datasets['test'], data_type=config.data_type, transform=test_transform)

    # Reconfigure batch-norm layers
    if config.balance:
        class_weight = torch.tensor(data_processor.class_weight, dtype=torch.float).to(local_rank)
        loss_function = nn.CrossEntropyLoss(weight=class_weight)
    else:
        loss_function = nn.CrossEntropyLoss()

    # Model (Task)
    model = SliceClassification(network=network)
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
        enable_wandb=config.enable_wandb,
        config=config
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
