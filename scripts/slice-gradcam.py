import json
import os
import sys
import time
import rich
import numpy as np
import pickle

import tqdm
import wandb

import torch
import torch.nn as nn

from configs.slice.classification import SliceClassificationConfig
from tasks.slice.classification import SliceClassification

from models.slice.backbone import ResNetBackbone
from models.slice.head import LinearClassifier

from datasets.brain import BrainProcessor, Brain
from datasets.slice.transforms import make_transforms

from utils.logging import get_rich_logger
from utils.gpu import set_gpu
import torch.optim as optim

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from monai.visualize import CAM, GradCAM, GradCAMpp, SmoothGrad
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize


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

hashs = ["2023-03-16_00-49-57"]

hash = hashs[0]
gpus = ["0"]
server = "main"

##
config = os.path.join(f'checkpoints/pet-SliceClassification/resnet50/{hash}/configs.json')
with open(config, 'rb') as fb:
    config = json.load(fb)
config = edict(config)

setattr(config, "server", server)
setattr(config, "gpus", gpus)
local_rank = 0
config.model_param = os.path.join(f'checkpoints/pet-SliceClassification/resnet50/{hash}/ckpt.last.pth.tar')

########
set_gpu(config)
np.random.seed(config.random_state)
torch.manual_seed(config.random_state)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_device(local_rank)

backbone = ResNetBackbone(name=config.backbone_type, in_channels=1)
classifier = LinearClassifier(in_channels=backbone.out_channels, num_classes=2)
if config.small_kernel:
    backbone._fix_first_conv()

backbone.load_weights_from_checkpoint(path=config.model_param, key='backbone')
classifier.load_weights_from_checkpoint(path=config.model_param, key='classifier')

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
                                                  num_slices=config.num_slices,
                                                  slice_range=config.slice_range,
                                                  prob=config.prob)

train_set = Brain(dataset=datasets['train'], data_type=config.data_type, transform=train_transform)
test_set = Brain(dataset=datasets['test'], data_type=config.data_type, transform=test_transform)

train_loader = DataLoader(dataset=train_set, batch_size=1, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=1, drop_last=False)

####
model = ModelViz(backbone, classifier, local_rank)
optimizer = optim.AdamW(model.parameters())

for mode, dset, loader in zip(['train', 'test'], [train_set, test_set], [train_loader, test_loader]):
    if mode == 'test':
        if type(hash) == tuple:
            path = f'gacm/{hash[0]}-{hash[1]}/{mode}'
        else:
            path = f'gcam/{hash}/{mode}'

        for layer in ['layer1']:

            for status in ['converter', 'nonconverter']:
                for reversed in ['original', 'reverse']:
                    for log in ['original', 'log']:
                        os.makedirs(os.path.join(path, layer, status, reversed, log), exist_ok=True)

            gcam = GradCAMpp(model, f'backbone.layers.{layer}')

            for batch in tqdm.tqdm(loader):
                idx = batch['idx'].item()
                xs = batch['x']

                logits = []
                for x in xs:
                    x = x.to(local_rank)
                    logit = model(x)
                    logits.append(logit.softmax(1))
                logits = torch.concat(logits, 0)
                logits = logits.mean(0).unsqueeze(0)
                logits = logits.softmax(1)
                confidence = "{:.3f}".format(logits[0, batch['y'].item()].item())


                if batch['y'].item() == logits.argmax().item():

                    pet_file = dset.pet[idx]
                    pet_id = pet_file.split('/')[-1].replace('.pkl', '')
                    with open(pet_file, 'rb') as fb:
                        pet = pickle.load(fb)
                    mask = pet <= 0
                    pet_id_ = pet_id.split('\\')[1]

                    # status & confidence
                    if batch['y'].item() == 0:
                        status = 'nonconverter'
                    else:
                        status = 'converter'

                    gcam_map_list = {
                        'original/original': None,
                        'original/log': None,
                        'reverse/original': None,
                        'reverse/log': None
                    }
                    for reversed in ['original', 'reverse']:

                        gcam_maps = []
                        gcam_maps_log = []

                        for x in xs:
                            optimizer.zero_grad()
                            x = x.to(local_rank)

                            gcam_map = gcam(x)
                            gcam_map = gcam_map.cpu().numpy()

                            if reversed == 'reverse':
                                gcam_map = np.abs(1 - gcam_map)

                            gcam_maps.append(gcam_map)
                            gcam_map_log = np.log(1 + gcam_map)
                            gcam_maps_log.append(gcam_map_log)

                        gcam_map_list[f'{reversed}/original'] = gcam_maps
                        gcam_map_list[f'{reversed}/log'] = gcam_maps_log

                    for k, v in gcam_map_list.items():
                        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
                        axs[0, 0].imshow(pet[72, :, :], cmap='binary')
                        axs[0, 2].imshow(pet[72, :, :], cmap='binary')
                        axs[1, 0].imshow(pet[:, 72, :], cmap='binary')
                        axs[1, 2].imshow(pet[:, 72, :], cmap='binary')
                        axs[2, 0].imshow(pet[:, :, 72], cmap='binary')
                        axs[2, 2].imshow(pet[:, :, 72], cmap='binary')
                        for i, img in enumerate(v):
                            img = img.squeeze()
                            if config.crop_size:
                                img = np.pad(img, 4)
                            img = resize(img, [145, 145])
                            axs[i, 1].imshow(img, cmap='jet')
                            axs[i, 2].imshow(img, cmap='jet', alpha=0.5)
                        plt.savefig(
                            os.path.join(path, layer, status, k) + f'/{pet_id_}-{confidence}.png',
                            dpi=300,
                            bbox_inches='tight'
                        )
                        plt.close()
