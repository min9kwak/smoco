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
from models.backbone.demonet import DemoNet
from models.head.classifier import LinearDemoClassifier

from datasets.brain import BrainProcessor, Brain
from datasets.transforms import make_transforms

from utils.logging import get_rich_logger
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

    config = DemoFinetuneConfig.parse_arguments()

    pretrained_file = os.path.join(config.pretrained_dir, "ckpt.last.pth.tar")
    setattr(config, 'pretrained_file', pretrained_file)

    pretrained_config = os.path.join(config.pretrained_dir, "configs.json")
    with open(pretrained_config, 'rb') as fb:
        pretrained_config = json.load(fb)

    # inherit pretrained configs
    # TODO: use data_parser() for fine_tune augmentation
    pretrained_config_names = [
        # data_parser
        'data_type', 'root', 'data_info', 'mci_only', 'n_splits', 'n_cv',
        'image_size', 'small_kernel', 'random_state',
        'intensity', 'crop', 'crop_size', 'rotate', 'flip', 'affine', 'blur', 'blur_std', 'prob',
        # model_parser
        'backbone_type', 'init_features', 'growth_rate', 'block_config', 'bn_size', 'dropout_rate',
        'arch', 'no_max_pool',
        # train
        'batch_size',
        # moco / supmoco
        'alphas',
        # others
        'task'
    ]

    for name in pretrained_config_names:
        if name in pretrained_config.keys():
            setattr(config, name, pretrained_config[name])

    config.task = config.task + f'_{config.finetune_type}_demo'

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

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_rich_logger(logfile=logfile)
        if config.enable_wandb:
            wandb.init(
                name=f'{config.backbone_type} : {config.hash}',
                project=f'sttr-{config.task}',
                config=config.__dict__
            )
    else:
        logger = None

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

    if config.freeze:
        backbone.freeze_weights()

    if config.freeze_bn:
        freeze_bn(backbone)

    # demo
    demo_encoder = DemoNet(in_channels=7, hidden=config.hidden)

    # classifier
    if config.crop_size:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.crop_size)
    else:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.image_size)

    classifier = LinearDemoClassifier(image_dims=out_dim, demo_dims=demo_encoder.out_features,
                                      num_classes=2, activation=activation)

    # load data
    data_processor = BrainProcessor(root=config.root,
                                    data_info=config.data_info,
                                    data_type=config.data_type,
                                    mci_only=config.mci_only,
                                    random_state=config.random_state)
    datasets = data_processor.process(n_splits=config.n_splits, n_cv=config.n_cv)

    # intensity normalization
    assert config.intensity in [None, 'scale', 'minmax', 'normalize']
    mean_std, min_max = (None, None), (None, None)
    if config.intensity == 'minmax':
        with open(os.path.join(config.root, 'labels/minmax.pkl'), 'rb') as fb:
            minmax_stats = pickle.load(fb)
            min_max = (minmax_stats[config.data_type]['min'], minmax_stats[config.data_type]['max'])
    else:
        pass

    train_transform, test_transform = make_transforms(image_size=config.image_size,
                                                      intensity=config.intensity,
                                                      min_max=min_max,
                                                      crop_size=config.crop_size,
                                                      rotate=config.rotate,
                                                      flip=config.flip,
                                                      affine=config.affine,
                                                      blur_std=config.blur_std,
                                                      prob=config.prob)

    finetune_transform = train_transform if config.finetune_trans == 'train' else test_transform
    train_set = Brain(dataset=datasets['train'], data_type=config.data_type, transform=finetune_transform)
    test_set = Brain(dataset=datasets['test'], data_type=config.data_type, transform=test_transform)

    # Reconfigure batch-norm layers
    if config.balance:
        class_weight = torch.tensor(data_processor.class_weight, dtype=torch.float).to(local_rank)
        loss_function = nn.CrossEntropyLoss(weight=class_weight)
    else:
        loss_function = nn.CrossEntropyLoss()

    # Model (Task)
    model = DemoClassification(backbone=backbone, demo_encoder=demo_encoder, classifier=classifier)
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
