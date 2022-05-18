import os

import torch
import pandas as pd
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MRIProcessor(object):
    def __init__(self,
                 root: str,
                 data_info: str,
                 random_state: int = 2022):
        self.root = root
        self.data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'MONTH': int, 'Conv': int})
        self.data_info = self.data_info[self.data_info['Conv'].isin([0, 1])]
        self.random_state = random_state

    def process(self, train_size):
        rids = self.data_info['RID'].unique()
        train_rids, test_rids = train_test_split(rids, train_size=train_size, random_state=self.random_state)

        train_files = self.data_info[self.data_info['RID'].isin(train_rids)]['MRI'].tolist()
        test_files = self.data_info[self.data_info['RID'].isin(test_rids)]['MRI'].tolist()

        brain_train = [os.path.join(self.root, 'FS7', f, 'mri/brain.mgz') for f in train_files]
        y_train = np.array([self.data_info[self.data_info['MRI'] == f]['Conv'].values[0] for f in train_files])

        brain_test = [os.path.join(self.root, 'FS7', f, 'mri/brain.mgz') for f in test_files]
        y_test = np.array([self.data_info[self.data_info['MRI'] == f]['Conv'].values[0] for f in test_files])

        assert all([os.path.isfile(b) for b in brain_train])
        assert all([os.path.isfile(b) for b in brain_test])

        datasets = {'train': {'path': brain_train, 'y': y_train},
                    'test': {'path': brain_test, 'y': y_test}}

        return datasets


class MRI(Dataset):

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
        # image = image / 255
        return image

    def slice_image(self, image):
        image = image[self.SLICE['xmin']:self.SLICE['xmax'],
                      self.SLICE['ymin']:self.SLICE['ymax'],
                      self.SLICE['zmin']:self.SLICE['zmax']]
        return image


if __name__ == '__main__':

    import seaborn as sns
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from monai.transforms import (
        AddChannel,
        Compose,
        RandRotate90,
        Resize,
        ScaleIntensity,
        EnsureType,
        ToTensor,
        RandFlip,
        RandZoom,
        CropForeground
    )

    processor = MRIProcessor(root='D:/data/ADNI',
                             data_info='labels/data_info.csv',
                             random_state=2022)
    datasets = processor.process(train_size=0.9)


    from torchvision.transforms import Normalize, ConvertImageDtype
    train_transform = Compose([ToTensor(),
                               AddChannel(),
                               Resize((96, 96, 96)),
                               ConvertImageDtype(torch.float32)])
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
