import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose, ToTensor, AddChannel
from datasets.mri import MRI


def compute_statistics(normalize_set):
    print('Start computing mean/std of the training dataset')
    normalize_transform = Compose([ToTensor(), AddChannel()])
    normalize_set = MRI(dataset=normalize_set, transform=normalize_transform, pin_memory=False)
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
    print('Computing mean/std is finished')

    return mean_, std_


