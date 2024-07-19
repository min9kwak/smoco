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
from torch.utils.data import DataLoader

from monai.visualize import (
    CAM, # Compute class activation map from the last fully-connected layers before the spatial pooling
    GradCAMpp,
    OcclusionSensitivity,
    SmoothGrad,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad,
)

import matplotlib.pyplot as plt
import seaborn as sns

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


sys.path.append('../')

# hashs = [("2022-07-02_08-00-31", "2022-07-03_12-54-32"),
#          ("2022-07-02_08-00-57", "2022-07-03_13-37-29"),
#          ("2022-07-02_09-38-52", "2022-07-03_13-33-23"),
#          ("2022-07-02_09-40-42", "2022-07-03_13-29-10"),
#          ("2022-07-02_11-17-38", "2022-07-03_13-25-05"),
#          ("2022-07-02_11-20-21", "2022-07-03_13-21-00"),
#          ("2022-07-02_17-15-14", "2022-07-03_13-16-54"),
#          ("2022-07-02_17-15-34", "2022-07-03_13-12-44"),
#          ("2022-07-02_18-53-46", "2022-07-03_13-08-35"),
#          ("2022-07-02_18-54-27", "2022-07-03_13-04-32")]

hashs = ["2023-02-19_21-21-17",
         "2023-02-19_23-37-31",
         "2023-02-20_01-53-17",
         "2023-02-20_04-08-26",
         "2023-02-20_06-24-10",
         "2023-02-20_08-40-33",
         "2023-02-20_10-55-48",
         "2023-02-20_13-10-42",
         "2023-02-20_15-26-26",
         "2023-02-20_17-42-38"]

hash = hashs[0]
gpus = ['0']
server = 'main'


########
config = edict()
config.server = server
config.gpus = gpus
local_rank = 0

if type(hash) == tuple:
    config.finetune_file = os.path.join(f'checkpoints/pet-supmoco/resnet/{hash[0]}/finetune/{hash[1]}/ckpt.last.pth.tar')
    finetune_config = os.path.join(f'checkpoints/pet-supmoco/resnet/{hash[0]}/finetune/{hash[1]}/configs.json')
else:
    config.finetune_file = os.path.join(f'checkpoints/pet/resnet/{hash}/ckpt.last.pth.tar')
    finetune_config = os.path.join(f'checkpoints/pet/resnet/{hash}/configs.json')

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

############
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
classifier = LinearClassifier(in_channels=out_dim, num_classes=2, activation=True) #########

backbone.load_weights_from_checkpoint(path=config.finetune_file, key='backbone')
classifier.load_weights_from_checkpoint(path=config.finetune_file, key='classifier')

# load finetune data
data_processor = BrainProcessor(root='D:/data/ADNI',
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

######
model = ModelViz(backbone, classifier, local_rank)

import torch.optim as optim
from skimage.transform import resize

optimizer = optim.AdamW(model.parameters())
model.eval()

for mode, dset, loader in zip(['train', 'test'], [train_set, test_set], [train_loader, test_loader]):

    if mode == 'test':
        if type(hash) == tuple:
            path = f'cam/{hash[0]}-{hash[1]}/{mode}'
        else:
            path = f'cam/{hash}/{mode}'

        for status in ['converter', 'nonconverter']:
            for reversed in ['original', 'reverse']:
                for log in ['original', 'log']:
                    os.makedirs(os.path.join(path, status, reversed, log), exist_ok=True)

        for batch in tqdm.tqdm(loader):
            x = batch['x'].to(local_rank)
            idx = batch['idx'].item()
            logit = model(x)
            logit = logit.detach()

            # correctly classified
            if batch['y'].item() == logit.argmax().item():

                optimizer.zero_grad()

                pet_file = dset.pet[idx]
                pet_id = pet_file.split('/')[-1].replace('.pkl', '')
                with open(pet_file, 'rb') as fb:
                    pet = pickle.load(fb)
                mask = pet <= 0

                # status & confidence
                if batch['y'].item() == 0:
                    status = 'nonconverter'
                else:
                    status = 'converter'
                confidence = "{:.3f}".format(logit.softmax(dim=1)[0, batch['y'].item()].item())

                # CAM
                cam_map_list = {
                    'original/original': None,
                    'original/log': None,
                    'reverse/original': None,
                    'reverse/log': None
                }

                cam = CAM(nn_module=model,
                          target_layers='classifier.layers.relu',
                          fc_layers='classifier.layers.linear')
                cam_map = cam(x)
                cam_map = cam_map[0][0].detach().cpu().numpy()
                cam_map_log = np.log(1 + cam_map)

                cam_map_rev = np.abs(1 - cam_map)
                cam_map_rev_log = np.log(1 + cam_map_rev)

                cam_map_list['original/original'] = cam_map
                cam_map_list['original/log'] = cam_map_log
                cam_map_list['reverse/original'] = cam_map_rev
                cam_map_list['reverse/log'] = cam_map_rev_log

                for k, v in cam_map_list.items():

                    # heatmap
                    img = resize(v, [145, 145, 145])

                    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
                    axs[0, 0].imshow(pet[72, :, :], cmap='binary')
                    axs[0, 1].imshow(img[72, :, :], cmap='jet')
                    axs[0, 2].imshow(pet[72, :, :], cmap='binary')
                    axs[0, 2].imshow(img[72, :, :], cmap='jet', alpha=0.5)

                    axs[1, 0].imshow(pet[:, 72, :], cmap='binary')
                    axs[1, 1].imshow(img[:, 72, :], cmap='jet')
                    axs[1, 2].imshow(pet[:, 72, :], cmap='binary')
                    axs[1, 2].imshow(img[:, 72, :], cmap='jet', alpha=0.5)

                    axs[2, 0].imshow(pet[:, :, 90], cmap='binary')
                    axs[2, 1].imshow(img[:, :, 90], cmap='jet')
                    axs[2, 2].imshow(pet[:, :, 90], cmap='binary')
                    axs[2, 2].imshow(img[:, :, 90], cmap='jet', alpha=0.5)
                    pet_id_ = pet_id.split('\\')[1]
                    plt.savefig(
                        os.path.join(path, status, k) + f'/{pet_id_}-{confidence}.png',
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()
