import torch
import torch.nn as nn
from collections import OrderedDict

from models.backbone.base import BackboneBase
from utils.initialization import initialize_weights


class DemoNet(BackboneBase):

    def __init__(self,
                 in_channels: int,
                 hidden: str) -> None:
        super(DemoNet, self).__init__(in_channels)

        self.in_channels = in_channels
        self.hidden = list(int(a) for a in hidden.split(','))

        self.out_features = self.hidden[-1]

        self.layers = None
        self.layers = self.make_layers(self)
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(self):
        layers = nn.Sequential()

        input_layer = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("input", nn.Linear(self.in_channels, self.hidden[0])),
                    ("bn", nn.BatchNorm1d(num_features=self.hidden[0])),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        layers.add_module("input", input_layer)

        if len(self.hidden) > 1:
            for i in range(len(self.hidden) - 1):
                block = nn.Sequential(
                    OrderedDict(
                        [
                            ("linear", nn.Linear(self.hidden[i], self.hidden[i + 1])),
                            ("bn", nn.BatchNorm1d(self.hidden[i + 1])),
                            ("relu", nn.ReLU(inplace=True)),
                        ]
                    )
                )
                layers.add_module(f"block{i}", block)
        return layers

    def forward(self, x: torch.FloatTensor):
        return self.layers(x)

if __name__ == '__main__':

    demonet_1 = DemoNet(7, "4")
    demonet_2 = DemoNet(7, "3,3")

    from datasets.brain import BrainProcessor, Brain
    processor = BrainProcessor(root='D:/data/ADNI',
                               data_info='labels/data_info.csv',
                               data_type='pet',
                               mci_only=True,
                               random_state=2021)
    datasets = processor.process(10, 0)

    from datasets.transforms import make_transforms

    train_transform, test_transform = make_transforms(image_size=72,
                                                      intensity='normalize',
                                                      rotate=True,
                                                      flip=True,
                                                      prob=0.5)
    train_set = Brain(dataset=datasets['train'], data_type='pet', transform=train_transform)

    from datasets.samplers import ImbalancedDatasetSampler
    from torch.utils.data import DataLoader

    train_sampler = ImbalancedDatasetSampler(dataset=train_set)
    train_loader = DataLoader(dataset=train_set, batch_size=16 // 2,
                              sampler=train_sampler, drop_last=True)

    for batch in train_loader:

        demo = batch['demo'].float()

        demo_feature_1 = demonet_1(demo)
        demo_feature_2 = demonet_2(demo)

        feature = torch.concat([demo_feature_1, demo_feature_2], dim=1)
