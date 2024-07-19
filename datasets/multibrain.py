import os
import pandas as pd
import numpy as np
import pickle

import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import class_weight


class MultiBrainProcessor(object):
    def __init__(self,
                 root: str,
                 data_info: str = 'labels/data_info_multi.csv',
                 random_state: int = 1):

        # TODO: select FDG or amyloid PET as argument
        self.root = root
        self.random_state = random_state
        self.demo_columns = ['PTGENDER (1=male, 2=female)', 'Age', 'PTEDUCAT',
                             'APOE Status', 'MMSCORE', 'CDGLOBAL', 'SUM BOXES']

        # for data split: data_info -> unlabeled, mri, fdg-pet
        self.data_info = pd.read_csv(os.path.join(self.root, data_info), converters={'RID': str, 'Conv': int})
        self.data_info = self.data_info.loc[self.data_info.IS_FILE]

        self.u_data_info = self.data_info.loc[self.data_info.Conv.isin([-1])].reset_index(drop=True)
        self.l_data_info = self.data_info.loc[self.data_info.Conv.isin([0, 1])].reset_index(drop=True)

        # use FDG --------
        self.complete_info = self.l_data_info.loc[~self.l_data_info.FDG.isna()].reset_index(drop=True)
        self.incomplete_info = self.l_data_info.loc[self.l_data_info.FDG.isna()].reset_index(drop=True)

    def process(self, n_splits=10, n_cv=0):

        assert n_splits in [5, 10]

        # CV split complete-modal
        rid = self.complete_info.RID.tolist()
        conv = self.complete_info.Conv.tolist()
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        assert 0 <= n_cv < n_splits

        # train-test split
        train_idx_list, test_idx_list = [], []
        for train_idx, test_idx in cv.split(X=rid, y=conv, groups=rid):
            train_idx_list.append(train_idx)
            self.append = test_idx_list.append(test_idx)
        train_idx, test_idx = train_idx_list[n_cv], test_idx_list[n_cv]

        complete_info_train = self.complete_info.iloc[train_idx].reset_index(drop=True)
        complete_info_test = self.complete_info.iloc[test_idx].reset_index(drop=True)

        # remove test rid from unlabeled and incomplete
        test_rid = list(set(complete_info_test.RID))
        self.u_data_info = self.u_data_info[~self.u_data_info.RID.isin(test_rid)].reset_index(drop=True)
        self.incomplete_info = self.incomplete_info[~self.incomplete_info.RID.isin(test_rid)].reset_index(drop=True)

        # train: complete_train + incomplete
        train_info = pd.concat([complete_info_train, self.incomplete_info]).reset_index(drop=True)

        # parse to make paths
        train_data = self.parse_data(train_info)
        test_data = self.parse_data(complete_info_test)
        u_train_data = self.parse_data(self.u_data_info)

        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(train_data['y']),
                                                              y=train_data['y'])

        datasets = {'train': train_data,
                    'test': test_data,
                    'u_train': u_train_data}

        return datasets

    def parse_data(self, data_info):
        mri_files = [self.str2mri(p) if type(p) == str else p for p in data_info.MRI]
        pet_files = [self.str2pet(p) if type(p) == str else p for p in data_info.FDG]
        # demo = data_info[self.demo_columns].values
        y = data_info.Conv.values
        # return dict(mri=mri_files, pet=pet_files, demo=demo, y=y)
        return dict(mri=mri_files, pet=pet_files, y=y)

    def str2mri(self, i):
        return os.path.join(self.root, 'template/FS7', f'{i}.pkl')

    def str2pet(self, i):
        # FDG
        return os.path.join(self.root, 'template/FDG', f'{i}.pkl')


class MultiBrainBase(Dataset):

    def __init__(self,
                 dataset: dict,
                 **kwargs):

        self.mri = dataset['mri']
        self.pet = dataset['pet']
        # self.demo = dataset['demo']
        self.y = dataset['y']

    def __len__(self):
        return len(self.y)

    @staticmethod
    def load_image(path):
        # TODO: if nan...
        with open(path, 'rb') as fb:
            image = pickle.load(fb)
        return image


class MultiBrain(MultiBrainBase):

    def __init__(self, dataset, data_type, transform, **kwargs):
        super().__init__(dataset, **kwargs)
        assert data_type in ['mri', 'multi']
        self.data_type = data_type
        self.transform = transform

    def __getitem__(self, idx):
        # TODO: include demographic info
        mri = self.load_image(path=self.mri[idx])
        if self.transform is not None:
            mri = self.transform(mri)
        y = self.y[idx]
        return dict(mri=mri, y=y, idx=idx)


if __name__ == '__main__':

    processor = MultiBrainProcessor(root='D:/data/ADNI',
                                    data_info='labels/data_info.csv',
                                    random_state=1)
    datasets = processor.process(n_splits=10, n_cv=0)

    train_set = MultiBrainBase(datasets['train'])
    test_set = MultiBrainBase(datasets['test'])


