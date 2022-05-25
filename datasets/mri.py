import os

import torch
import pandas as pd
import numpy as np
import pickle
import nibabel as nib

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


class MRIProcessor(object):
    def __init__(self,
                 root: str,
                 data_info: str,
                 random_state: int = 2022):

        self.root = root
        self.data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'MONTH': int, 'Conv': int})

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

        train_files = self.data_info[self.data_info['RID'].isin(train_rids)]['MRI'].tolist()
        test_files = self.data_info[self.data_info['RID'].isin(test_rids)]['MRI'].tolist()
        u_train_files = self.u_data_info['MRI'].tolist()

        brain_train = [os.path.join(self.root, 'FS7', f, 'mri/brain.mgz') for f in train_files]
        y_train = np.array([self.data_info[self.data_info['MRI'] == f]['Conv'].values[0] for f in train_files])

        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_train),
                                                              y=y_train)

        brain_test = [os.path.join(self.root, 'FS7', f, 'mri/brain.mgz') for f in test_files]
        y_test = np.array([self.data_info[self.data_info['MRI'] == f]['Conv'].values[0] for f in test_files])

        u_brain_train = [os.path.join(self.root, 'FS7', f, 'mri/brain.mgz') for f in u_train_files]
        u_y_train = np.array([self.u_data_info[self.u_data_info['MRI'] == f]['Conv'].values[0] for f in u_train_files])

        assert all([os.path.isfile(b) for b in brain_train])
        assert all([os.path.isfile(b) for b in brain_test])
        assert all([os.path.isfile(b) for b in u_brain_train])

        datasets = {'train': {'path': brain_train, 'y': y_train},
                    'test': {'path': brain_test, 'y': y_test},
                    'u_train': {'path': u_brain_train, 'y': u_y_train}}

        return datasets


class MRI(Dataset):

    # SLICE = {'xmin': 30, 'xmax': 222,
    #          'ymin': 50, 'ymax': 242,
    #          'zmin': 35, 'zmax': 227}

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

        image = nib.load(path)
        image = nib.as_closest_canonical(image)
        image = image.get_fdata()
        image = self.slice_image(image)

        return image

    def slice_image(self, image):
        image = image[self.SLICE['xmin']:self.SLICE['xmax'],
                      self.SLICE['ymin']:self.SLICE['ymax'],
                      self.SLICE['zmin']:self.SLICE['zmax']]
        return image


if __name__ == '__main__':

    import time
    import seaborn as sns
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from datasets.transforms import compute_statistics, make_transforms

    processor = MRIProcessor(root='D:/data/ADNI',
                             data_info='labels/data_info.csv',
                             random_state=2022)
    datasets = processor.process(train_size=0.9)

    s1 = time.time()
    mean_std = compute_statistics(DATA=MRI, normalize_set=datasets['train'])
    e1 = time.time()
    print(e1 - s1)
    train_transform, test_transform = make_transforms(image_size=96,
                                                      intensity='normalize',
                                                      mean_std=mean_std,
                                                      rotate=True,
                                                      flip=True,
                                                      zoom=True,
                                                      blur=True,
                                                      blur_std=0.1,
                                                      prob=1.0)

    train_set = datasets['train']
    train_set = MRI(dataset=train_set, transform=train_transform, pin_memory=False)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=False)

    for batch in train_loader:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        sns.heatmap(batch['x'][0, 0, 48, :, :], cmap='binary', ax=axs[0])
        sns.heatmap(batch['x'][0, 0, :, 48, :], cmap='binary', ax=axs[1])
        sns.heatmap(batch['x'][0, 0, :, :, 48], cmap='binary', ax=axs[2])
        plt.show()
        break

    # TODO: check loading speed of MRI (mgz and pkl)
    import glob
    import pickle
    import os
    DATA_DIR = "D:/data/ADNI/FS7/m127S0925L111010M615TCF/mri"
    image_file = os.path.join(DATA_DIR, 'brain.mgz')

    import time
    s1 = time.time()
    image = nib.load(image_file)
    image = nib.as_closest_canonical(image)
    image = image.get_fdata()
    e1 = time.time()
    print(e1-s1)

    s2 = time.time()
    with open(image_file.replace('.mgz', '.pkl'), 'rb') as f:
        image = pickle.load(f)
    e2 = time.time()
    print(e2 - s2)
