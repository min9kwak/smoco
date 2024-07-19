import os
import sys
import json

import pandas as pd
import numpy as np
import pickle

import pandas as pd
import tqdm

import torch
import torch.nn as nn

from models.backbone.base import calculate_out_features
from models.backbone.densenet import DenseNetBackbone
from models.backbone.resnet import build_resnet_backbone
from models.head.projector import MLPHead
from models.head.classifier import LinearClassifier

from datasets.brain import BrainProcessor, Brain, BrainMoCo
from datasets.transforms import make_transforms, compute_statistics

from utils.gpu import set_gpu

from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Subset

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.metrics import classification_result, regression_result


sys.path.append('../')

gpus = ['0']
server = 'main'

hashs = [("2022-07-02_08-00-31", "2022-07-03_13-41-32"),
         ("2022-07-02_08-00-57", "2022-07-03_13-37-29"),
         ("2022-07-02_09-38-52", "2022-07-03_13-33-23"),
         ("2022-07-02_09-40-42", "2022-07-03_13-29-10"),
         ("2022-07-02_11-17-38", "2022-07-03_13-25-05"),
         ("2022-07-02_11-20-21", "2022-07-03_13-21-00"),
         ("2022-07-02_17-15-14", "2022-07-03_13-16-54"),
         ("2022-07-02_17-15-34", "2022-07-03_13-12-44"),
         ("2022-07-02_18-53-46", "2022-07-03_13-08-35"),
         ("2022-07-02_18-54-27", "2022-07-03_13-04-32")]

# def (hash)
def probing_randomforest(hash):

    config = edict()
    config.server = server
    config.gpus = gpus
    local_rank = 0

    config.finetune_file = os.path.join(f'checkpoints/pet-supmoco/resnet/{hash[0]}/ckpt.last.pth.tar')
    finetune_config = os.path.join(f'checkpoints/pet-supmoco/resnet/{hash[0]}/configs.json')
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

    if config.crop:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.crop_size)
    else:
        out_dim = calculate_out_features(backbone=backbone, in_channels=1, image_size=config.image_size)
    projector = MLPHead(out_dim, config.projector_dim)

    backbone.load_weights_from_checkpoint(path=config.finetune_file, key='backbone')
    projector.load_weights_from_checkpoint(path=config.finetune_file, key='head')

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

    train_set = Brain(dataset=datasets['train'], data_type=config.data_type, transform=test_transform)
    test_set = Brain(dataset=datasets['test'], data_type=config.data_type, transform=test_transform)

    train_loader = DataLoader(dataset=train_set, batch_size=16, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=16, drop_last=False)

    backbone.to(local_rank)
    projector.to(local_rank)

    backbone.eval()
    projector.eval()

    train = {'h': [], 'z': [], 'y': [], 'demo': [], 'mc': [], 'volume': []}
    test = {'h': [], 'z': [], 'y': [], 'demo': [], 'mc': [], 'volume': []}

    with torch.no_grad():

        print('Extracting training set...')
        for batch in train_loader:
            x = batch['x'].to(local_rank)
            h = backbone(x)
            z = projector(h)
            h = nn.AdaptiveAvgPool3d(1)(h)
            h = torch.squeeze(h)

            train['h'] += [h.detach().cpu().numpy()]
            train['z'] += [z.detach().cpu().numpy()]
            train['y'] += [batch['y'].detach().cpu().numpy()]
            train['demo'] += [batch['demo'].detach().cpu().numpy()]
            train['mc'] += [batch['mc'].detach().cpu().numpy()]
            train['volume'] += [batch['volume'].detach().cpu().numpy()]

        print('Extracting testing set...')
        for batch in test_loader:
            x = batch['x'].to(local_rank)
            h = backbone(x)
            z = projector(h)
            h = nn.AdaptiveAvgPool3d(1)(h)
            h = torch.squeeze(h)

            test['h'] += [h.detach().cpu().numpy()]
            test['z'] += [z.detach().cpu().numpy()]
            test['y'] += [batch['y'].detach().cpu().numpy()]
            test['demo'] += [batch['demo'].detach().cpu().numpy()]
            test['mc'] += [batch['mc'].detach().cpu().numpy()]
            test['volume'] += [batch['volume'].detach().cpu().numpy()]

    for k, v in train.items():
        try:
            train[k] = np.concatenate(v)
        except:
            train[k] = np.array(v)

    for k, v in test.items():
        try:
            test[k] = np.concatenate(v)
        except:
            test[k] = np.array(v)

    Y_train = np.concatenate([train['demo'], train['mc'].reshape(-1, 1), train['volume'].reshape(-1, 1)], 1)
    Y_train = pd.DataFrame(Y_train, columns=['gender', 'age', 'education', 'apoe', 'mmscore', 'mc', 'volume'])

    Y_test = np.concatenate([test['demo'], test['mc'].reshape(-1, 1), test['volume'].reshape(-1, 1)], 1)
    Y_test = pd.DataFrame(Y_test, columns=['gender', 'age', 'education', 'apoe', 'mmscore', 'mc', 'volume'])


    result_ = {}

    for target_name in Y_train:
        y_train = Y_train[target_name].values
        y_test = Y_test[target_name].values
        if target_name in ['gender', 'apoe']:
            # classification
            rf = RandomForestClassifier()
            rf.fit(train['z'], y_train)
            y_test_pred = rf.predict_proba(test['z'])
            result_[f'{target_name}-original'] = classification_result(y_test, y_test_pred, False)
            result_[f'{target_name}-adjusted'] = classification_result(y_test, y_test_pred, True)
        else:
            # regression
            rf = RandomForestRegressor()
            rf.fit(train['z'], y_train)
            y_test_pred = rf.predict(test['z'])
            result_[f'{target_name}'] = regression_result(y_test, y_test_pred)

    return result_

result = []
for hash in tqdm.tqdm(hashs):
    result_ = probing_randomforest(hash)
    result.append(result_)

target_names = result[0].keys()

for target_name in target_names:
    print(target_name, '-'*10)
    for metric in result[0][target_name].keys():
        print(metric, np.mean([res[target_name][metric] for res in result]))

import shap
shap.DeepExplainer()