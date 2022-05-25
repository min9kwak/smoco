import os
import pickle

import torch
import nibabel as nib
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

class PETProcessor(object):
    def __init__(self,
                 root: str,
                 data_info: str,
                 random_state: int = 2022):
        self.root = root
        self.data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'MONTH': int, 'Conv': int})
        self.data_info = self.data_info[self.data_info['Conv'].isin([0, 1])]
        self.data_info = self.data_info[~self.data_info.PET.isna()]
        self.random_state = random_state

    def process(self, train_size):
        rids = self.data_info['RID'].unique()
        train_rids, test_rids = train_test_split(rids, train_size=train_size, random_state=self.random_state)

        self.train_rids = train_rids
        self.test_rids = test_rids

        train_info = self.data_info[self.data_info['RID'].isin(train_rids)]
        test_info = self.data_info[self.data_info['RID'].isin(test_rids)]

        train_files = [os.path.join(self.root, 'PUP_FBP', row.PET, f'pet_proc/{row.PET}_SUVR.pkl')
                       for _, row in train_info.iterrows()]
        y_train = train_info['Conv'].values

        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_train),
                                                              y=y_train)

        test_files = [os.path.join(self.root, 'PUP_FBP', row.PET, f'pet_proc/{row.PET}_SUVR.pkl')
                      for _, row in test_info.iterrows()]
        y_test = test_info['Conv'].values

        assert all([os.path.isfile(a) for a in train_files])
        assert all([os.path.isfile(a) for a in test_files])

        datasets = {'train': {'path': train_files, 'y': y_train},
                    'test': {'path': test_files, 'y': y_test}}

        return datasets


class PET(Dataset):

    SLICE = {'xmin': 30, 'xmax': 222,
             'ymin': 30, 'ymax': 222,
             'zmin': 30, 'zmax': 222}

    def __init__(self,
                 dataset: dict,
                 transform: object = None,
                 pin_memory: bool = True,
                 **kwargs):

        self.transform = transform
        self.pin_memory = pin_memory

        self.paths = dataset['path']
        self.y = dataset['y']

        self.images = []
        if self.pin_memory:
            self.images = [self.load_image(path) for path in self.paths]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.y[idx]
        if self.pin_memory:
            img = self.images[idx]
        else:
            img = self.load_image(path)

        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=y, idx=idx)

    def load_image(self, path):
        with open(path, 'rb') as f:
            image = pickle.load(f)
        image = self.slice_image(image)
        return image

    def slice_image(self, image):
        image = image[self.SLICE['xmin']:self.SLICE['xmax'],
                      self.SLICE['ymin']:self.SLICE['ymax'],
                      self.SLICE['zmin']:self.SLICE['zmax']]
        return image


if __name__ == '__main__':

    import numpy as np
    for random_state in [2021, 2022, 2023, 2024, 2025]:
        processor = PETProcessor(root='D:/data/ADNI',
                                 data_info='labels/data_info.csv',
                                 random_state=random_state)
        datasets = processor.process(train_size=0.9)
        test_set = datasets['test']

        print(np.bincount(test_set['y']))
    bins = np.bincount(processor.data_info['Conv'].values)
    bins[0]/np.sum(bins)
