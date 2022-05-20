import time

import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, AddChannel, RandRotate90, Resize, ScaleIntensity, ToTensor, RandFlip, RandZoom,
    NormalizeIntensity, RandGaussianNoise
)
from torchvision.transforms import ConvertImageDtype, Normalize


def make_transforms(image_size: int = 96,
                    intensity: str = 'normalize',
                    mean_std: tuple = (None, None),
                    rotate: bool = True,
                    flip: bool = True,
                    zoom: bool = True,
                    blur: bool = True,
                    blur_std: float = 0.1,
                    prob: float = 0.2):

    base_transform = [ToTensor(),
                      AddChannel(),
                      Resize((image_size, image_size, image_size))]
    if intensity is None:
        pass
    elif intensity == 'scale':
        base_transform.insert(1, ScaleIntensity())
    elif intensity == 'normalize':
        assert all(mean_std)
        base_transform.insert(1, Normalize(*mean_std))
    else:
        raise NotImplementedError

    train_transform, test_transform = base_transform.copy(), base_transform.copy()

    if rotate:
        train_transform.append(RandRotate90(prob=prob))
    if flip:
        train_transform.append(RandFlip(prob=prob))
    if zoom:
        train_transform.append(RandZoom(prob=prob))
    if blur:
        train_transform.append(RandGaussianNoise(prob=prob, std=blur_std))

    train_transform.append(ConvertImageDtype(torch.float32))
    test_transform.append(ConvertImageDtype(torch.float32))

    return Compose(train_transform), Compose(test_transform)


def compute_statistics(DATA, normalize_set):

    print('Start computing mean/std of the training dataset')

    start_time = time.time()
    normalize_transform = Compose([ToTensor(), AddChannel()])
    normalize_set = DATA(dataset=normalize_set, transform=normalize_transform, pin_memory=False)
    normalize_loader = DataLoader(normalize_set, batch_size=16, shuffle=False, drop_last=False)

    count = 0
    first_moment, second_moment = torch.zeros(1), torch.zeros(1)
    for batch in normalize_loader:
        x = batch['x'].float()
        b, _, h, d, w = x.shape
        num_pixels = b * h * d * w

        sum_ = torch.sum(x)
        sum_of_square_ = torch.sum(x ** 2)

        first_moment = first_moment + sum_
        second_moment = second_moment + sum_of_square_
        count = count + num_pixels

    first_moment = first_moment / count
    second_moment = second_moment / count

    mean_ = first_moment
    std_ = torch.sqrt(second_moment - first_moment ** 2)
    mean_ = mean_.item()
    std_ = std_.item()
    print(f'Mean and std values are computed in {time.time() - start_time:.2f} seconds')

    return (mean_, std_)
