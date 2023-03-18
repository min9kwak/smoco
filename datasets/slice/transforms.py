import time
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, AddChannel, RandRotate, RandRotate90, Resize, ScaleIntensity, ToTensor, RandFlip, RandZoom, RandAffine,
    RandSpatialCrop, NormalizeIntensity, RandGaussianNoise, Transform, CenterSpatialCrop,
    AddChannel
)
from torchvision.transforms import ConvertImageDtype, Normalize
from monai.utils.enums import TransformBackends
from monai.config import DtypeLike
from monai.utils import convert_data_type

from datasets.transforms import MinMax


class RandomSlices(Transform):

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, num_slices: int = 4, image_size: int = 72, slice_range: float = 0.15):
        self.num_slices = num_slices
        self.image_size = image_size

        m = int(image_size * slice_range)
        center = image_size // 2
        self.slice_range = (center - m, center + m + 1)

    def __call__(self, img, *args, **kwargs):

        ret = []
        for i in range(self.num_slices):
            dim = random.choice(range(3))
            point = random.choice(range(*self.slice_range))
            if dim == 0:
                slice = img[:, point, :, :]
            elif dim == 1:
                slice = img[:, :, point, :]
            elif dim == 2:
                slice = img[:, :, :, point]
            else:
                raise ValueError
            ret.append(slice)
        return ret


class FixedSlices(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, image_size: int = 72):
        self.image_size = image_size
        self.center = image_size // 2

    def __call__(self, img, *args, **kwargs):

        ret = []
        ret.append(img[:, self.center, :, :])
        ret.append(img[:, :, self.center, :])
        ret.append(img[:, :, :, self.center])

        return ret


class SingleSlices(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, image_size: int = 72, slice_view: str = 'sagittal'):
        self.image_size = image_size
        self.center = image_size // 2
        assert slice_view in ['sagittal', 'coronal', 'axial']
        self.slice_view = slice_view

    def __call__(self, img, *args, **kwargs):
        ret = []
        if self.slice_view == 'sagittal':
            ret.append(img[:, self.center, :, :])
        elif self.slice_view == 'coronal':
            ret.append(img[:, :, self.center, :])
        elif self.slice_view == 'axial':
            ret.append(img[:, :, :, self.center])
        return ret


def make_transforms(image_size: int = 72,
                    intensity: str = 'scale',
                    min_max: tuple = (None, None),
                    crop_size: int = None,
                    rotate: bool = True,
                    flip: bool = True,
                    affine: bool = True,
                    blur_std: float = 0.1,
                    train_slices: str = 'random',
                    num_slices: int = 4,
                    slice_range: float = 0.15,
                    prob: float = 0.2):

    base_transform = [ToTensor(),
                      AddChannel(),
                      Resize((image_size, image_size, image_size))]
    if intensity is None:
        pass
    elif intensity == 'scale':
        base_transform.insert(1, ScaleIntensity())
    elif intensity == 'normalize':
        base_transform.insert(1, NormalizeIntensity(nonzero=True))
    elif intensity == 'minmax':
        assert all(min_max)
        base_transform.insert(1, MinMax(*min_max))
    else:
        raise NotImplementedError

    train_transform, test_transform = base_transform.copy(), base_transform.copy()

    if crop_size:
        # train_transform.append(RandSpatialCrop(roi_size=(cropsize, cropsize, cropsize), random_size=False))
        train_transform.append(CenterSpatialCrop(roi_size=(crop_size, crop_size, crop_size)))
        test_transform.append(CenterSpatialCrop(roi_size=(crop_size, crop_size, crop_size)))

    if rotate:
        train_transform.append(RandRotate90(prob=prob))

    if flip:
        train_transform.append(RandFlip(prob=prob))

    if affine:
        import warnings
        warnings.filterwarnings("ignore")
        train_transform.append(RandAffine(rotate_range=(-2.0, 2.0),
                                          translate_range=(-4.0, 4.0),
                                          scale_range=(0.95, 1.05),
                                          prob=prob))

    if blur_std:
        train_transform.append(RandGaussianNoise(prob=prob, std=blur_std))

    if crop_size is not None:
        slice_size = crop_size
    else:
        slice_size = image_size

    if train_slices == 'random':
        train_transform.append(RandomSlices(num_slices=num_slices,
                                            image_size=slice_size,
                                            slice_range=slice_range))
    elif train_slices == 'fixed':
        train_transform.append(FixedSlices(image_size=slice_size))
    elif train_slices in ['sagittal', 'coronal', 'axial']:
        train_transform.append(SingleSlices(image_size=slice_size, slice_view=train_slices))
    else:
        raise ValueError

    if train_slices in ['random', 'fixed']:
        test_transform.append(FixedSlices(image_size=slice_size))
    elif train_slices in ['sagittal', 'coronal', 'axial']:
        test_transform.append(SingleSlices(image_size=slice_size, slice_view=train_slices))
    else:
        raise ValueError

    train_transform.append(ConvertImageDtype(torch.float32))
    test_transform.append(ConvertImageDtype(torch.float32))

    return Compose(train_transform), Compose(test_transform)


if __name__ == '__main__':

    from datasets.brain import BrainProcessor
    processor = BrainProcessor(root='D:/data/ADNI',
                               data_info='labels/data_info.csv',
                               data_type='pet',
                               mci_only=True,
                               random_state=2021)
    datasets = processor.process(10, 0)

    train_transform, test_transform = make_transforms(image_size=72,
                                                      intensity='scale',
                                                      crop_size=64,
                                                      affine=False,
                                                      blur_std=None)
    from datasets.brain import Brain
    train_set = Brain(dataset=datasets['train'], data_type='pet', transform=train_transform)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, drop_last=True)
    for batch in train_loader:
        x = torch.concat(batch['x'])
        y = batch['y'].repeat(4)
        x.cuda()
        y.cuda()
        break