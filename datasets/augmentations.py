import random

import torch
from monai.transforms import (
    Compose, AddChannel, RandRotate, RandRotate90, Resize, ScaleIntensity, ToTensor, RandFlip, RandZoom, RandAffine,
    RandSpatialCrop, NormalizeIntensity, RandGaussianNoise, Transform, CenterSpatialCrop, RandSpatialCrop,
    EnsureChannelFirst
)
from torchvision.transforms import ConvertImageDtype


def make_randaugment(image_size: int = 72,
                     intensity: str = 'normalize',
                     k: int = 2):

    base_transform = [ToTensor(), EnsureChannelFirst(), Resize((image_size, image_size, image_size))]

    # scaling
    if intensity is None:
        pass
    elif intensity == 'scale':
        base_transform.insert(1, ScaleIntensity())
    elif intensity == 'normalize':
        base_transform.insert(1, NormalizeIntensity(nonzero=True))
    else:
        raise NotImplementedError

    train_transform, test_transform = base_transform.copy(), base_transform.copy()

    train_transform.append(RandAugment(k=k))
    train_transform.append(ConvertImageDtype(torch.float32))
    test_transform.append(ConvertImageDtype(torch.float32))

    print(train_transform)
    print(test_transform)

    return Compose(train_transform), Compose(test_transform)


class RandAugment(object):
    def __init__(self, k: int = 2):
        self.k = k
        self.transform = MultipleRandomChoice(
            [
                # TODO: check augmentations...
            ],
            k=self.k
        )

    def __call__(self, img):
        return self.transform(img)


class MultipleRandomChoice(object):
    """Apply a total of `k` randomly selected transforms."""

    def __init__(self, transforms: list or tuple, k: int = 5, verbose: bool = False):
        self.transforms = transforms
        self.k = k
        self.verbose = verbose

    def __call__(self, img):
        transforms = random.choices(self.transforms, k=self.k)
        for t in transforms:
            if self.verbose:
                print(str(t), end='\n')
            img = t(img)
        return img
