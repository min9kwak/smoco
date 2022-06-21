# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from datasets.depr.mri import MRI, MRIProcessor
from datasets.depr.pet import PET, PETProcessor

from configs.depr.densenet import DenseNetConfig

from models.network.vit import UnimodalViT
from models.network.densenet import UnimodalDenseNet
from models.network.resnet import build_unimodal_resnet

from tasks.classification import Classification

from utils.logging import get_rich_logger
from utils.gpu import set_gpu
from datasets.transforms import make_transforms, compute_statistics


def main():
    """Main function for single/distributed linear classification."""

    # config = VitUniConfig.parse_arguments()
    config = DenseNetConfig.parse_arguments()
    # config = ResNetUniConfig.parse_arguments()

    config.task = config.data_type

    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    # str -> list arguments
    if config.backbone == 'densenet':
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
            name=f'{config.backbone} : {config.hash}',
            project=f'sttr-{config.task}',
            config=config.__dict__
        )

    # Networks
    if config.backbone == 'vit':
        network = UnimodalViT(in_channels=1,
                              img_size=(config.image_size, config.image_size, config.image_size),
                              patch_size=(config.patch_size, config.patch_size, config.patch_size),
                              hidden_size=config.hidden_size,
                              mlp_dim=config.mlp_dim,
                              num_layers=config.num_layers,
                              num_heads=config.num_heads,
                              pos_embed=config.pos_embed,
                              num_classes=2,
                              dropout_rate=config.dropout_rate)
    elif config.backbone == 'densenet':
        network = UnimodalDenseNet(in_channels=1,
                                   out_channels=2,
                                   init_features=config.init_features,
                                   growth_rate=config.growth_rate,
                                   block_config=config.block_config,
                                   bn_size=config.bn_size,
                                   dropout_rate=config.dropout_rate)
    elif config.backbone == 'resnet':
        network = build_unimodal_resnet(arch=config.arch,
                                        no_max_pool=config.no_max_pool,
                                        in_channels=1,
                                        num_classes=2)

    # load data
    if config.task == 'mri':
        PROCESSOR = MRIProcessor
        DATA = MRI
    elif config.task == 'pet':
        PROCESSOR = PETProcessor
        DATA = PET
    else:
        raise NotImplementedError

    data_processor = PROCESSOR(root=config.root,
                               data_info=config.data_info,
                               random_state=config.random_state)
    datasets = data_processor.process(train_size=config.train_size)

    if config.intensity == 'normalize':
        mean_std = compute_statistics(DATA=DATA, normalize_set=datasets['train'])
    else:
        mean_std = (None, None)

    train_transform, test_transform = make_transforms(image_size=config.image_size,
                                                      intensity=config.intensity,
                                                      mean_std=mean_std,
                                                      rotate=config.rotate,
                                                      flip=config.flip,
                                                      zoom=config.zoom,
                                                      blur=config.blur,
                                                      blur_std=config.blur_std,
                                                      prob=config.prob)

    train_set = DATA(dataset=datasets['train'], transform=train_transform, pin_memory=config.pin_memory)
    test_set = DATA(dataset=datasets['test'], transform=test_transform, pin_memory=config.pin_memory)

    # Reconfigure batch-norm layers
    loss_function = nn.CrossEntropyLoss()

    # Model (Task)
    model = Classification(network=network)
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
