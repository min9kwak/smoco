import os
import sys
import json
import time
import rich
import numpy as np
import pickle
import wandb
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.finetune import FinetuneConfig
from tasks.classification import Classification

from models.backbone.base import calculate_out_features
from models.backbone.densenet import DenseNetBackbone
from models.backbone.resnet import build_resnet_backbone
from models.head.projector import MLPHead
from models.head.classifier import LinearClassifier

from datasets.brain import BrainProcessor, Brain, BrainMoCo
from datasets.transforms import make_transforms, compute_statistics

from utils.logging import get_rich_logger
from utils.gpu import set_gpu

from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

import nibabel as nib
from skimage.transform import resize

from copy import deepcopy
import glob

from monai.visualize import GradCAMpp


class ModelViz(nn.Module):
    def __init__(self, backbone, classifier, local_rank):
        super(ModelViz, self).__init__()
        self.local_rank = local_rank
        self.backbone = backbone
        self.classifier = classifier
        self._build_model(self.backbone, self.classifier)

    def _build_model(self, backbone, classifier):
        self.backbone = backbone
        self.classifier = classifier

        self.backbone.to(self.local_rank)
        self.classifier.to(self.local_rank)

        self.backbone.eval()
        self.classifier.eval()

    def forward(self, x):
        logits = self.classifier(self.backbone(x))
        return logits

checkpoint_root = 'D:/Dropbox/Projects/Alzheimer Disease/checkpoints/pet/resnet'
hashs = os.listdir(checkpoint_root)
gpus = ['0']
server = 'main'

##
for hash in hashs:

    # Set config
    config = edict()
    config.server = server
    config.gpus = gpus
    local_rank = 0

    config.finetune_file = os.path.join(checkpoint_root, hash, 'ckpt.last.pth.tar')
    finetune_config = os.path.join(checkpoint_root, hash, 'configs.json')
    with open(finetune_config, 'rb') as fb:
        finetune_config = json.load(fb)

    finetune_config_names = [
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
        'task', 'projector_dim'
    ]

    for name in finetune_config_names:
        if name in finetune_config.keys():
            setattr(config, name, finetune_config[name])

    #########################################
    set_gpu(config)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)

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

    if config.crop_size:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.crop_size)
    else:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.image_size)
    classifier = LinearClassifier(in_channels=out_dim, num_classes=2, activation=activation)

    backbone.load_weights_from_checkpoint(path=config.finetune_file, key='backbone')
    classifier.load_weights_from_checkpoint(path=config.finetune_file, key='classifier')

    # load finetune data
    data_processor = BrainProcessor(root=config.root,
                                    data_info=config.data_info,
                                    data_type=config.data_type,
                                    mci_only=config.mci_only,
                                    random_state=config.random_state)
    datasets = data_processor.process(n_splits=config.n_splits, n_cv=config.n_cv)

    # intensity normalization
    assert config.intensity in [None, 'scale', 'minmax']
    mean_std, min_max = (None, None), (None, None)
    if config.intensity is None:
        pass
    elif config.intensity == 'scale':
        pass
    elif config.intensity == 'minmax':
        with open(os.path.join(config.root, 'labels/minmax.pkl'), 'rb') as fb:
            minmax_stats = pickle.load(fb)
            min_max = (minmax_stats[config.data_type]['min'], minmax_stats[config.data_type]['max'])
    else:
        raise NotImplementedError

    train_transform, test_transform = make_transforms(image_size=config.image_size,
                                                      intensity=config.intensity,
                                                      min_max=min_max,
                                                      crop_size=config.crop_size,
                                                      rotate=config.rotate,
                                                      flip=config.flip,
                                                      affine=config.affine,
                                                      blur_std=config.blur_std,
                                                      prob=config.prob)

    #########################################
    train_set = Brain(dataset=datasets['train'], data_type=config.data_type, transform=test_transform)
    test_set = Brain(dataset=datasets['test'], data_type=config.data_type, transform=test_transform)

    train_loader = DataLoader(dataset=train_set, batch_size=1, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, drop_last=False)

    ###############
    model = ModelViz(backbone=backbone, classifier=classifier, local_rank=local_rank)

    import torch.optim as optim

    optimizer = optim.AdamW(model.parameters())

    ##############
    # save individual
    mode = 'train'
    dset = train_set
    loader = train_loader

    for layer in ['layer1', 'layer2']:

        gcam = GradCAMpp(model, f'backbone.{layer}')
        path = f'gcam/{layer}/{hash}/{mode}'
        os.makedirs(path + '-converter', exist_ok=True)
        os.makedirs(path + '-nonconverter', exist_ok=True)
        os.makedirs(path + '-converter-reverse', exist_ok=True)
        os.makedirs(path + '-nonconverter-reverse', exist_ok=True)

        average_map = {
            'converter': {'map': [], 'confidence': []},
            'nonconverter': {'map': [], 'confidence': []}
        }

        for batch in tqdm.tqdm(loader):

            x = batch['x'].to(local_rank)
            idx = batch['idx'].item()

            logit = model(x)
            logit = logit.detach()
            confidence = logit.softmax(dim=1)[0, batch['y'].item()].item()

            # correctly classified
            if batch['y'].item() == logit.argmax().item():

                for reverse in ['-reverse', '']:

                    optimizer.zero_grad()

                    gcam_map = gcam(x)
                    gcam_map = gcam_map.cpu().numpy()[0][0]
                    if reverse == '-reverse':
                        gcam_map = np.abs(1 - gcam_map)

                    if not np.isnan(gcam_map).any():
                        # status
                        if batch['y'].item() == 0:
                            status = 'nonconverter'
                        else:
                            status = 'converter'
                        # heatmap
                        pet_file = dset.pet[idx]
                        pet_id = pet_file.split('/')[-1].replace('.pkl', '')
                        pet_id = pet_id.replace('PUP_FBP\\', '')
                        with open(pet_file, 'rb') as fb:
                            pet = pickle.load(fb)

                        mask = pet <= 0

                        gcam_map = resize(gcam_map, [145, 145, 145])
                        average_map[status]['map'].append(gcam_map)
                        average_map[status]['confidence'].append(confidence)
                        gcam_map[mask] = np.nan
                        confidence_ = "{:.3f}".format(confidence)

                        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                        axs[0, 0].imshow(pet[72, :, :], cmap='binary')
                        axs[0, 1].imshow(gcam_map[72, :, :], cmap='jet')

                        axs[1, 0].imshow(pet[:, 72, :], cmap='binary')
                        axs[1, 1].imshow(gcam_map[:, 72, :], cmap='jet')

                        axs[2, 0].imshow(pet[:, :, 90], cmap='binary')
                        axs[2, 1].imshow(gcam_map[:, :, 90], cmap='jet')
                        plt.savefig(
                            path + '-' + status + reverse + f'/{pet_id}-{confidence_}.png',
                            dpi=300,
                            bbox_inches='tight'
                        )
                        plt.close()

        for status in ['converter', 'nonconverter']:
            gcam_map = [m * c for m, c in zip(average_map[status]['map'], average_map[status]['confidence'])]
            gcam_map = np.sum(gcam_map, axis=0) / np.sum(average_map['converter']['confidence'])

            for reverse in ['', '-reverse']:

                if reverse == '-reverse':
                    gcam_map = np.abs(1 - gcam_map)

                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                axs[0, 0].imshow(pet[72, :, :], cmap='binary')
                axs[0, 1].imshow(gcam_map[72, :, :], cmap='jet')

                axs[1, 0].imshow(pet[:, 72, :], cmap='binary')
                axs[1, 1].imshow(gcam_map[:, 72, :], cmap='jet')

                axs[2, 0].imshow(pet[:, :, 90], cmap='binary')
                axs[2, 1].imshow(gcam_map[:, :, 90], cmap='jet')
                plt.savefig(
                    path + '-' + status + reverse + f'/average.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
