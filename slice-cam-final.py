import os
import json
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from models.backbone.base import calculate_out_features
from models.backbone.densenet import DenseNetBackbone
from models.backbone.resnet import build_resnet_backbone
from models.head.classifier import LinearClassifier

from datasets.brain import BrainProcessor
from datasets.transforms import make_transforms
from utils.gpu import set_gpu

from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from skimage.transform import resize
from monai.visualize import GradCAMpp

'''
FBP002S1268L060111M4-0.929          2023-02-19_21-21-17\test-converter-reverse
FBP021S0626L081810M4-0.989-center   2023-02-19_23-37-31\train
FBP032S2247L041911E4-0.986          2023-02-19_21-21-17\test-nonconverter-reverse
FBP126S0680L071813N4-0.991-center   2023-02-20_01-53-17\train-nonconverter-reverse
'''


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
        self.classifier.eval`()

    def forward(self, x):
        logits = self.classifier(self.backbone(x))
        return logits


# def draw_figure(hash, gpus, server, mode, selected_id, local_rank):
hash_mode_id_list = [
    ["2023-02-19_21-21-17", "test", "FBP002S1268L060111M4"],
    ["2023-02-19_23-37-31", "train", "FBP021S0626L081810M4"],
    ["2023-02-19_21-21-17", "test", "FBP032S2247L041911E4"],
    ["2023-02-20_01-53-17", "test", "FBP126S0680L071813N4"],
]

for hash_mode_id in hash_mode_id_list:

    hash, mode, pet_id = hash_mode_id

    gpus = ['0']
    server = 'main'
    checkpoint_root = 'D:/Dropbox/Projects/Alzheimer Disease/checkpoints/pet/resnet'

    ##
    # for hash_mode_id in hash_mode_id_list:

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

    ###############
    model = ModelViz(backbone=backbone, classifier=classifier, local_rank=local_rank)


    optimizer = optim.AdamW(model.parameters())

    layer = 'layer1'
    fig_count = 0

    gcam = GradCAMpp(model, f'backbone.{layer}')
    path = f'gcam-fin/'
    os.makedirs(path + 'pet/', exist_ok=True)
    os.makedirs(path + 'mri/', exist_ok=True)

    # PET
    pet_path = os.path.join(f'D:/data/ADNI/template/PUP_FBP/{pet_id}.pkl')
    with open(pet_path, 'rb') as fb:
        pet = pickle.load(fb)

    # MRI
    data_info = pd.read_csv("D:/data/ADNI/labels/data_info.csv")
    mri_id = data_info.loc[data_info['PET'] == pet_id]['MRI'].item()
    mri_path = f"D:/data/ADNI/template/FS7/{mri_id}.pkl"
    with open(mri_path, 'rb') as fb:
        mri = pickle.load(fb)

    x = test_transform(pet).unsqueeze(0)
    x = x.to(local_rank)

    optimizer.zero_grad()

    gcam_map = gcam(x)
    gcam_map = gcam_map.cpu().numpy()[0][0]
    gcam_map = np.abs(1 - gcam_map)
    gcam_map = np.log(1 + gcam_map)

    mask = pet <= 0

    mri[mask] = np.nan
    pet[mask] = np.nan
    gcam_map = resize(gcam_map, [145, 145, 145])
    gcam_map[mask] = np.nan

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    axs[0, 0].imshow(np.rot90(mri[72, :, :]), cmap='binary')
    axs[0, 1].imshow(np.rot90(pet[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(pet[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(gcam_map[72, :, :]), cmap='jet', alpha=0.3)

    axs[1, 0].imshow(np.rot90(mri[:, 72, :]), cmap='binary')
    axs[1, 1].imshow(np.rot90(pet[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(pet[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(gcam_map[:, 72, :]), cmap='jet', alpha=0.3)

    axs[2, 0].imshow(np.rot90(mri[:, :, 90]), cmap='binary')
    axs[2, 1].imshow(np.rot90(pet[:, :, 90]), cmap='binary')
    axs[2, 2].imshow(np.rot90(pet[:, :, 90]), cmap='binary')
    axs[2, 2].imshow(np.rot90(gcam_map[:, :, 90]), cmap='jet', alpha=0.3)

    for ax in axs:
        for ax_ in ax:
            ax_.axis('off')
    plt.tight_layout()
    plt.savefig(
        path + f'pet/{pet_id}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    axs[0, 0].imshow(np.rot90(mri[72, :, :]), cmap='binary')
    axs[0, 1].imshow(np.rot90(pet[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(pet[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(gcam_map[72, :, :]), cmap='jet', alpha=0.3)

    axs[1, 0].imshow(np.rot90(mri[:, 72, :]), cmap='binary')
    axs[1, 1].imshow(np.rot90(pet[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(pet[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(gcam_map[:, 72, :]), cmap='jet', alpha=0.3)

    axs[2, 0].imshow(np.rot90(mri[:, :, 72]), cmap='binary')
    axs[2, 1].imshow(np.rot90(pet[:, :, 72]), cmap='binary')
    axs[2, 2].imshow(np.rot90(pet[:, :, 72]), cmap='binary')
    axs[2, 2].imshow(np.rot90(gcam_map[:, :, 72]), cmap='jet', alpha=0.3)

    for ax in axs:
        for ax_ in ax:
            ax_.axis('off')
    plt.tight_layout()
    plt.savefig(
        path + f'pet/{pet_id}-center.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    axs[0, 0].imshow(np.rot90(mri[72, :, :]), cmap='binary')
    axs[0, 1].imshow(np.rot90(pet[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(mri[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(gcam_map[72, :, :]), cmap='jet', alpha=0.3)

    axs[1, 0].imshow(np.rot90(mri[:, 72, :]), cmap='binary')
    axs[1, 1].imshow(np.rot90(pet[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(mri[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(gcam_map[:, 72, :]), cmap='jet', alpha=0.3)

    axs[2, 0].imshow(np.rot90(mri[:, :, 90]), cmap='binary')
    axs[2, 1].imshow(np.rot90(pet[:, :, 90]), cmap='binary')
    axs[2, 2].imshow(np.rot90(mri[:, :, 90]), cmap='binary')
    axs[2, 2].imshow(np.rot90(gcam_map[:, :, 90]), cmap='jet', alpha=0.3)

    for ax in axs:
        for ax_ in ax:
            ax_.axis('off')
    plt.tight_layout()
    plt.savefig(
        path + f'mri/{pet_id}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    axs[0, 0].imshow(np.rot90(mri[72, :, :]), cmap='binary')
    axs[0, 1].imshow(np.rot90(pet[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(mri[72, :, :]), cmap='binary')
    axs[0, 2].imshow(np.rot90(gcam_map[72, :, :]), cmap='jet', alpha=0.3)

    axs[1, 0].imshow(np.rot90(mri[:, 72, :]), cmap='binary')
    axs[1, 1].imshow(np.rot90(pet[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(mri[:, 72, :]), cmap='binary')
    axs[1, 2].imshow(np.rot90(gcam_map[:, 72, :]), cmap='jet', alpha=0.3)

    axs[2, 0].imshow(np.rot90(mri[:, :, 72]), cmap='binary')
    axs[2, 1].imshow(np.rot90(pet[:, :, 72]), cmap='binary')
    axs[2, 2].imshow(np.rot90(mri[:, :, 72]), cmap='binary')
    axs[2, 2].imshow(np.rot90(gcam_map[:, :, 72]), cmap='jet', alpha=0.3)

    for ax in axs:
        for ax_ in ax:
            ax_.axis('off')
    plt.tight_layout()
    plt.savefig(
        path + f'mri/{pet_id}-center.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
