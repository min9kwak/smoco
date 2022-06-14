import os
import pickle

import numpy as np
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
                 mci_only: bool = False,
                 segment: str = 'left_hippocampus',
                 random_state: int = 2022):

        self.root = root
        self.data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'MONTH': int, 'Conv': int})
        self.data_info = self.data_info[~self.data_info.PET.isna()]
        if mci_only:
            self.data_info = self.data_info.loc[self.data_info.MCI == 1]
        assert segment in ['global', 'left_hippocampus', 'right_hippocampus', 'hippocampus']
        self.segment = segment

        # remove abnormal MRI
        with open(os.path.join(root, 'labels/mri_abnormal.pkl'), 'rb') as fb:
            mri_abnormal = pickle.load(fb)
        self.data_info = self.data_info.loc[~self.data_info.MRI.isin(mri_abnormal)]

        # unlabeled and labeled
        self.u_data_info = self.data_info[self.data_info['Conv'].isin([-1])]
        self.data_info = self.data_info[self.data_info['Conv'].isin([0, 1])]

        self.random_state = random_state

    def process(self, train_size):

        rids = self.data_info['RID'].unique()
        train_rids, test_rids = train_test_split(rids, train_size=train_size, random_state=self.random_state)

        self.train_rids = train_rids
        self.test_rids = test_rids
        self.u_data_info = self.u_data_info[~self.u_data_info['RID'].isin(test_rids)]

        train_info = self.data_info[self.data_info['RID'].isin(train_rids)]
        test_info = self.data_info[self.data_info['RID'].isin(test_rids)]

        train_files = [os.path.join(self.root, f'segment/PUP_FBP/{self.segment}', f'{row.PET}.pkl')
                       for _, row in train_info.iterrows()]
        y_train = train_info['Conv'].values

        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_train),
                                                              y=y_train)

        test_files = [os.path.join(self.root, f'segment/PUP_FBP/{self.segment}', f'{row.PET}.pkl')
                      for _, row in test_info.iterrows()]
        y_test = test_info['Conv'].values

        u_train_files = [os.path.join(self.root, f'segment/PUP_FBP/{self.segment}', f'{row.PET}.pkl')
                         for _, row in self.u_data_info.iterrows()]
        u_y_train = self.u_data_info['Conv'].values

        datasets = {'train': {'path': train_files, 'y': y_train},
                    'test': {'path': test_files, 'y': y_test},
                    'u_train': {'path': u_train_files, 'y': u_y_train}}

        return datasets


class PETBase(Dataset):

    def __init__(self,
                 dataset: dict,
                 pin_memory: bool = True,
                 **kwargs):

        self.pin_memory = pin_memory

        self.paths = dataset['path']
        self.y = dataset['y']

        self.images = []
        if self.pin_memory:
            self.images = [self.load_image(path) for path in self.paths]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raise NotImplementedError

    def load_image(self, path):
        with open(path, 'rb') as f:
            image = pickle.load(f)
        return image


class PET(PETBase):

    def __init__(self, dataset, pin_memory, transform, **kwargs):
        super().__init__(dataset, pin_memory, **kwargs)
        self.transform = transform

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


class PETMoCo(PETBase):

    def __init__(self, dataset, pin_memory, query_transform, key_transform, **kwargs):
        super().__init__(dataset, pin_memory, **kwargs)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.y[idx]
        if self.pin_memory:
            img = self.images[idx]
        else:
            img = self.load_image(path)

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)

        return dict(x1=x1, x2=x2, y=y, idx=idx)


if __name__ == '__main__':

    processor = PETProcessor(root='D:/data/ADNI',
                             data_info='labels/data_info.csv',
                             mci_only=False,
                             segment='global',
                             random_state=2022)
    datasets = processor.process(train_size=0.9)

    from torch.utils.data import Dataset, DataLoader
    train_set = PET(dataset=datasets['train'], pin_memory=False, transform=None)
    train_loader = DataLoader(train_set)

    for batch in train_loader:
        image = batch['x']
        image = image.squeeze().numpy()

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        sns.heatmap(image[48, :, :], cmap='binary', ax=axs[0], vmax=2)
        sns.heatmap(image[:, 48, :], cmap='binary', ax=axs[1], vmax=2)
        sns.heatmap(image[:, :, 48], cmap='binary', ax=axs[2], vmax=2)
        plt.tight_layout()
        plt.show()
        break
