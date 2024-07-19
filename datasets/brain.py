import os
import pandas as pd
import numpy as np
import pickle

import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler


class BrainProcessor(object):
    def __init__(self,
                 root: str,
                 data_info: str = 'labels/data_info.csv',
                 data_type: str = 'pet',
                 mci_only: bool = False,
                 add_apoe: bool = False,
                 add_volume: bool = False,
                 random_state: int = 2022):

        self.root = root
        # TODO: include multi... define __getitem__ = self.something
        assert data_type in ['mri', 'pet']
        self.data_type = data_type
        self.random_state = random_state
        self.demo_columns = ['PTGENDER (1=male, 2=female)', 'Age', 'PTEDUCAT', 'MMSCORE']
        self.add_apoe = add_apoe
        self.add_volume = add_volume

        if self.add_apoe:
            self.demo_columns = self.demo_columns + ['APOE Status']

        self.data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'Conv': int})
        self.data_info = self.data_info.loc[self.data_info.IS_FILE]

        # preprocess MC and Hippocampus Volume
        self._preprocess_mc_hippo()

        # preprocess demo
        self._preprocess_demo()

        # include hippocampus volume for simplicity
        if self.add_volume:
            self.demo_columns = self.demo_columns + ['Volume']

        self.num_demo_columns = len(self.demo_columns)

        # data filtering
        if mci_only:
            self.data_info = self.data_info.loc[self.data_info.MCI == 1]

        if self.data_type == 'mri':
            self.data_info = self.data_info[~self.data_info.MRI.isna()]
        else:
            self.data_info = self.data_info[~self.data_info.PET.isna()]

        # unlabeled and labeled
        self.u_data_info = self.data_info[self.data_info['Conv'].isin([-1])].reset_index(drop=True)
        self.data_info = self.data_info[self.data_info['Conv'].isin([0, 1])].reset_index(drop=True)

    def process(self, n_splits=10, n_cv=0):

        # prepare
        rid = self.data_info.RID.tolist()
        conv = self.data_info.Conv.tolist()
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        assert 0 <= n_cv < n_splits

        # train-test split
        train_idx_list, test_idx_list = [], []
        for train_idx, test_idx in cv.split(X=rid, y=conv, groups=rid):
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)

        train_idx, test_idx = train_idx_list[n_cv], test_idx_list[n_cv]
        train_info = self.data_info.iloc[train_idx].reset_index(drop=True)
        test_info = self.data_info.iloc[test_idx].reset_index(drop=True)

        # filter rids in unlabeled data
        test_rid = list(set(test_info.RID))
        self.u_data_info = self.u_data_info[~self.u_data_info.RID.isin(test_rid)]
        u_train_info = self.u_data_info.reset_index(drop=True)

        # Demographic & Clinical Data Preprocessing
        scaler = MinMaxScaler()
        scaler.fit(train_info[self.demo_columns])

        train_info[self.demo_columns] = scaler.transform(train_info[self.demo_columns])
        test_info[self.demo_columns] = scaler.transform(test_info[self.demo_columns])
        u_train_info[self.demo_columns] = scaler.transform(u_train_info[self.demo_columns])

        # parse to make paths
        train_data = self.parse_data(train_info)
        test_data = self.parse_data(test_info)
        u_train_data = self.parse_data(u_train_info)

        # set class weight
        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(train_data['y']),
                                                              y=train_data['y'])

        datasets = {'train': train_data,
                    'test': test_data,
                    'u_train': u_train_data}

        return datasets

    def _preprocess_mc_hippo(self):

        mc_table = pd.read_excel(os.path.join(self.root, 'labels/AV45_FBP_SUVR.xlsx'), sheet_name='list_id_SUVR_RSF')
        mri_table = pd.read_csv(os.path.join(self.root, 'labels/MRI_BAI_features.csv'))

        # MC
        mc_table = mc_table.rename(columns={'ID': 'PET'})
        self.data_info = pd.merge(left=self.data_info, right=mc_table[['PET', 'MC']], how='left', on='PET')
        self.data_info['MC'] = 53.6 * self.data_info['MC'] - 43.2

        # Hippocampus Volume
        mri_table['MRI'] = mri_table['Filename'].str.split('/', expand=True).iloc[:, 1]
        mri_table['Volume'] = mri_table['Left-Hippocampus'] + mri_table['Right-Hippocampus']
        self.data_info = pd.merge(left=self.data_info, right=mri_table[['MRI', 'Volume']], how='left', on='MRI')

    def _preprocess_demo(self):

        data_demo = self.data_info[['RID', 'Conv'] + self.demo_columns]

        # 1. Gender
        if 'PTGENDER (1=male, 2=female)' in self.demo_columns:
            data_demo_ = data_demo.copy()
            data_demo_['PTGENDER (1=male, 2=female)'] = data_demo_['PTGENDER (1=male, 2=female)'] - 1
            data_demo = data_demo_.copy()

        # 2. APOE Status
        if 'APOE Status' in self.demo_columns:
            data_demo_ = data_demo.copy()
            data_demo_.loc[:, 'APOE Status'] = data_demo['APOE Status'].fillna('NC')
            data_demo_['APOE Status'] = [0 if a == 'NC' else 1 for a in data_demo_['APOE Status'].values]
            data_demo = data_demo_.copy()

        # 3. Others
        cols = ['MMSCORE', 'CDGLOBAL', 'SUM BOXES']
        cols = [c for c in cols if c in self.demo_columns]
        records = data_demo.groupby('Conv').mean()[cols].to_dict()

        data_demo_ = data_demo.copy()
        for col in cols:
            nan_index = data_demo_.index[data_demo[col].isna()]
            for i in nan_index:
                value = records[col][data_demo_.loc[i, 'Conv']]
                data_demo_.loc[i, col] = value
        data_demo = data_demo_.copy()
        self.data_info[self.demo_columns] = data_demo[self.demo_columns]

    def parse_data(self, data_info):
        mri_files = [self.str2mri(p) if type(p) == str else p for p in data_info.MRI]
        pet_files = [self.str2pet(p) if type(p) == str else p for p in data_info.PET]
        demo = data_info[self.demo_columns].values
        mc = data_info['MC'].values
        volume = data_info['Volume'].values
        y = data_info.Conv.values
        return dict(mri=mri_files, pet=pet_files, demo=demo, mc=mc, volume=volume, y=y)

    def str2mri(self, i):
        return os.path.join(self.root, 'template/FS7', f'{i}.pkl')

    def str2pet(self, i):
        return os.path.join(self.root, 'template/PUP_FBP', f'{i}.pkl')


class BrainBase(Dataset):

    def __init__(self,
                 dataset: dict,
                 data_type: str,
                 **kwargs):
        self.mri = dataset['mri']
        self.pet = dataset['pet']
        self.demo = dataset['demo']
        self.mc = dataset['mc']
        self.volume = dataset['volume']
        self.y = dataset['y']

        assert data_type in ['mri', 'pet']
        self.data_type = data_type

        if self.data_type == 'mri':
            self.paths = self.mri
        elif self.data_type == 'pet':
            self.paths = self.pet
        else:
            raise ValueError

    def __len__(self):
        return len(self.y)

    @staticmethod
    def load_image(path):
        with open(path, 'rb') as fb:
            image = pickle.load(fb)
        return image


class Brain(BrainBase):

    def __init__(self, dataset, data_type, transform, **kwargs):
        super().__init__(dataset, data_type, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        img = self.load_image(path=self.paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        demo = self.demo[idx]
        mc = self.mc[idx]
        volume = self.volume[idx]
        y = self.y[idx]
        return dict(x=img, demo=demo, mc=mc, volume=volume, y=y, idx=idx)


class BrainMoCo(BrainBase):

    def __init__(self, dataset, data_type, query_transform, key_transform, **kwargs):
        super().__init__(dataset, data_type, **kwargs)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx):
        img = self.load_image(path=self.paths[idx])
        x1 = self.query_transform(img)
        x2 = self.key_transform(img)
        demo = self.demo[idx]
        mc = self.mc[idx]
        volume = self.volume[idx]
        y = self.y[idx]
        return dict(x1=x1, x2=x2, demo=demo, mc=mc, volume=volume, y=y, idx=idx)


class BrainMixUp(BrainBase):

    def __init__(self, dataset, data_type, transform, alpha, **kwargs):
        super().__init__(dataset, data_type, **kwargs)
        self.transform = transform
        self.alpha = alpha

    def __getitem__(self, idx):

        # one-hot label
        y = torch.zeros(2)
        y[self.y[idx]] = 1.0

        # original image
        img = self.load_image(path=self.paths[idx])
        if self.transform is not None:
            img = self.transform(img)

        # mixup image
        idx_mix = random.randint(0, self.__len__() - 1)
        y_mix = torch.zeros(2)
        y_mix[self.y[idx_mix]] = 1.0

        img_mix = self.load_image(path=self.paths[idx_mix])
        if self.transform is not None:
            img_mix = self.transform(img_mix)

        lam = np.random.beta(self.alpha, self.alpha)
        img_mix = lam * img + (1 - lam) * img_mix
        y_mix = lam * y + (1 - lam) * y_mix

        demo = self.demo[idx]

        return dict(x=img, x_mix=img_mix, y=y, y_mix=y_mix, demo=demo, idx=idx, idx_mix=idx_mix)


if __name__ == '__main__':

    processor = BrainProcessor(root='D:/data/ADNI',
                               data_info='labels/data_info.csv',
                               data_type='pet',
                               mci_only=True,
                               add_volume=True,
                               random_state=2021,
                               )
    datasets = processor.process(10, 0)

    from datasets.transforms import make_transforms
    train_transform, test_transform = make_transforms(image_size=72,
                                                      intensity='normalize',
                                                      rotate=True,
                                                      flip=True,
                                                      prob=0.5)
    train_set = BrainMoCo(dataset=datasets['train'], data_type='pet',
                          query_transform=train_transform, key_transform=train_transform)

    from datasets.samplers import ImbalancedDatasetSampler
    from torch.utils.data import DataLoader
    train_sampler = ImbalancedDatasetSampler(dataset=train_set)
    train_loader = DataLoader(dataset=train_set, batch_size=16 // 2,
                              sampler=train_sampler, drop_last=True)
    for batch in train_loader:
        break