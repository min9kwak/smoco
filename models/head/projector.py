import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.head.base import HeadBase
from utils.initialization import initialize_weights


class LinearHead(HeadBase):
    def __init__(self, in_channels: int, num_features: int, dropout: float = 0.0):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output features.
        """
        super(LinearHead, self).__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        self.dropout = dropout
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_features=self.num_features,
            dropout=self.dropout,
        )
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(in_channels: int, num_features: int, dropout: float = 0.0):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool3d(1)),
                    ('flatten', nn.Flatten(1)),
                    ('dropout', nn.Dropout(p=dropout)),
                    ('linear', nn.Linear(in_channels, num_features))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPHead(HeadBase):
    def __init__(self, in_channels: int, num_features: int):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output units.
        """
        super(MLPHead, self).__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_features=self.num_features
        )

    @staticmethod
    def make_layers(in_channels: int, num_features: int):

        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool3d(1)),
                    ('flatten', nn.Flatten(1)),
                    ('linear1', nn.Linear(in_channels, in_channels, bias=False)),
                    ('bnorm1', nn.BatchNorm1d(in_channels)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('linear2', nn.Linear(in_channels, num_features, bias=True))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
