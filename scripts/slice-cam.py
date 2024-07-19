import json
import os
import numpy as np
import pickle

import tqdm
import torch
import torch.nn as nn

from datasets.brain import BrainProcessor, Brain
from datasets.slice.transforms import make_transforms
from models.slice.resnet import resnet50, resnet18
from utils.gpu import set_gpu
import torch.optim as optim

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from monai.visualize import (
    GradCAM,
    GradCAMpp,
    OcclusionSensitivity,
)
import matplotlib.pyplot as plt
from skimage.transform import resize


# models
def draw_figure(hash, gpus, server, figure_max):
    
    # create path
    path = f'slice/{hash}'
    
    # config
    config = os.path.join(f'checkpoints/pet-SliceClassification/resnet50/{hash}/configs.json')
    with open(config, 'rb') as fb:
        config = json.load(fb)
    config = edict(config)

    config.model_param = os.path.join(f'checkpoints/pet-SliceClassification/resnet50/{hash}/ckpt.last.pth.tar')
    
    config.gpus = gpus
    config.server = server    

    set_gpu(config)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    
    # network
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

    ckpt = torch.load(config.model_param, map_location='cpu')
    network.load_state_dict(ckpt['network'])

    network.to(local_rank)
    network.eval()

    # data
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

    train_set = Brain(dataset=datasets['train'], data_type=config.data_type, transform=test_transform)
    test_set = Brain(dataset=datasets['test'], data_type=config.data_type, transform=test_transform)

    train_loader = DataLoader(dataset=train_set, batch_size=1, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, drop_last=False)
    
    
    # single model
    optimizer = optim.AdamW(network.parameters())
    
    # explaining method
    gradcam1 = GradCAM(network, 'layer1')
    gradcam2 = GradCAM(network, 'layer2')
    gradcampp1 = GradCAMpp(network, 'layer1')
    gradcampp2 = GradCAMpp(network, 'layer2')
    occsens4 = OcclusionSensitivity(network, mask_size=4, n_batch=1, verbose=False)

    methods = [gradcam1, gradcam2, gradcampp1, gradcampp2, occsens4]

    # RUN
    for mode, dset, loader in zip(['test', 'train'], [test_set, train_set], [test_loader, train_loader]):
        os.makedirs(os.path.join(path, mode, 'converter'), exist_ok=True)
        os.makedirs(os.path.join(path, mode, 'nonconverter'), exist_ok=True)

        figure_cnt = 0

        for batch in tqdm.tqdm(loader):

            if figure_cnt == figure_max:
                continue

            # load data
            idx = batch['idx'].item()
            xs = batch['x']
            

            # predict
            logits = []
            for x in xs:
                x = x.to(local_rank)
                logit = network(x)
                logits.append(logit)
            logits = torch.concat(logits, 0)
            logits = logits.mean(0).unsqueeze(0)
            logits = logits.softmax(1)
            confidence = "{:.3f}".format(logits[0, batch['y'].item()].item())

            if batch['y'].item() == logits.argmax().item():

                # open PET scan
                pet_file = dset.pet[idx]
                pet_id = pet_file.split('/')[-1].replace('.pkl', '')
                with open(pet_file, 'rb') as fb:
                    pet = pickle.load(fb)
                mask = pet <= 0
                pet[mask] = np.nan

                # PET scan slice
                sagittal = pet[72, :, :]
                coronal  = pet[:, 72, :]
                axial    = pet[:, :, 72]
                images = [sagittal, coronal, axial]

                # status & confidence
                if batch['y'].item() == 0:
                    status = 'nonconverter'
                else:
                    status = 'converter'

                fig, axs = plt.subplots(len(xs)*2, len(methods)+1, figsize=((len(methods)+1)*5, len(xs)*5*2))

                for i, x in enumerate(xs):
                    # PET image
                    image = images[i]
                    for j in range(len(methods)+1):
                        axs[i, j].imshow(image, cmap='binary')
                        axs[i + len(xs), j].imshow(image, cmap='binary')

                    # maps
                    x = x.to(local_rank)
                    for j, method in enumerate(methods):
                        optimizer.zero_grad()
                        m = method(x)
                        if method.__class__.__name__ == 'OcclusionSensitivity':
                            m = m[0][0][batch['y'], ...][0]
                        
                        m = m.cpu().numpy().squeeze()
                        m_rev = np.abs(1 - m)                        
                        
                        m = resize(m, [145, 145])
                        m_rev = resize(m_rev, [145, 145])
                        
                        m[np.isnan(image)] = np.nan
                        m_rev[np.isnan(image)] = np.nan
                        
                        axs[i, j+1].imshow(m, cmap='jet', alpha=0.5)
                        axs[i + len(xs), j+1].imshow(m_rev, cmap='jet', alpha=0.5)

                pet_id_ = pet_id.split('\\')[-1]
                plt.savefig(
                    os.path.join(path, mode, status) + f'/{pet_id_}-{confidence}.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()

                figure_cnt = figure_cnt + 1
                

hashs = [
    "2023-03-28_20-48-43", # random 100 145
    # "2023-03-28_16-36-04", # fixed 50 145
    # "2023-03-28_15-40-43", # random 50 145
    "2023-03-27_02-40-00", # random 50 72
    # "2023-03-26_21-41-40", # fixed 50 72
]


gpus = ["0"]
server = "main"
local_rank = 0

for hash in hashs:
    draw_figure(hash, gpus, server, 40)
