import matplotlib.pyplot as plt
import seaborn as sns

import os

import torch
import tqdm
from nilearn import surface
import nibabel as nib
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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

        train_info = self.data_info[self.data_info['RID'].isin(train_rids)]
        test_info = self.data_info[self.data_info['RID'].isin(test_rids)]

        train_files = [(os.path.join(self.root, 'FS7', row.MRI, 'mri/brainmask.mgz'),
                        os.path.join(self.root, 'PUP_FBP', row.PET, f'pet_proc/{row.PET}_SUVR.nii.gz'))
                       for _, row in train_info.iterrows()]
        y_train = train_info['Conv'].values

        test_files = [(os.path.join(self.root, 'FS7', row.MRI, 'mri/brainmask.mgz'),
                       os.path.join(self.root, 'PUP_FBP', row.PET, f'pet_proc/{row.PET}_SUVR.nii.gz'))
                      for _, row in test_info.iterrows()]
        y_test = test_info['Conv'].values

        assert all([os.path.isfile(a) for a, b in train_files])
        assert all([os.path.isfile(b) for a, b in train_files])
        assert all([os.path.isfile(a) for a, b in test_files])
        assert all([os.path.isfile(b) for a, b in test_files])

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
        mask, image = path
        mask, image = nib.load(mask), nib.load(image)
        mask, image = nib.as_closest_canonical(mask), nib.as_closest_canonical(image)
        mask, image = mask.get_fdata(), image.get_fdata().squeeze()
        mask = (mask == 0)
        image[mask] = 0
        image = self.slice_image(image)
        # image = image / 255
        return image

    def slice_image(self, image):
        image = image[self.SLICE['xmin']:self.SLICE['xmax'],
                      self.SLICE['ymin']:self.SLICE['ymax'],
                      self.SLICE['zmin']:self.SLICE['zmax']]
        return image


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import DataLoader
    from monai.transforms import AddChannel, Compose, Resize, ToTensor

    processor = PETProcessor(root='D:/data/ADNI',
                             data_info='labels/data_info.csv',
                             random_state=2022)
    datasets = processor.process(train_size=0.9)


    from torchvision.transforms import ConvertImageDtype
    train_transform = Compose([ToTensor(),
                               AddChannel(),
                               Resize((96, 96, 96)),
                               ConvertImageDtype(torch.float32)])
    train_set = datasets['train']
    train_set = PET(dataset=train_set, transform=train_transform, pin_memory=False)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=False)

    for batch in train_loader:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        sns.heatmap(batch['x'][0, 0, 48, :, :], cmap='binary', ax=axs[0])
        sns.heatmap(batch['x'][0, 0, :, 48, :], cmap='binary', ax=axs[1])
        sns.heatmap(batch['x'][0, 0, :, :, 48], cmap='binary', ax=axs[2])
        plt.show()
        break
