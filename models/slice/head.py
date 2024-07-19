# -*- coding: utf-8 -*-

import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.initialization import initialize_weights


class HeadBase(nn.Module):
    def __init__(self, output_size: int):
        super(HeadBase, self).__init__()
        assert isinstance(output_size, int)

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False

    def save_weights(self, path: str):
        """Save weights to a file with weights only."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """Load weights from a file with weights only."""
        self.load_state_dict(torch.load(path))

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])


class FCHeadBase(HeadBase):
    def __init__(self, input_size: int, output_size: int):
        super(FCHeadBase, self).__init__(output_size)
        assert isinstance(input_size, int), "Number of input units."


class GAPHeadBase(HeadBase):
    def __init__(self, in_channels: int, output_size: int):
        super(GAPHeadBase, self).__init__(output_size)
        assert isinstance(in_channels, int), "Number of output feature maps of backbone."


class LinearHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int, dropout: float = 0.0):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output features.
        """
        super(LinearHead, self).__init__(in_channels, num_features)

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
                    ('gap', nn.AdaptiveAvgPool2d(1)),
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


class MLPHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output units.
        """
        super(MLPHead, self).__init__(in_channels, num_features)

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
                    ('gap', nn.AdaptiveAvgPool2d(1)),
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


class LinearClassifier(LinearHead):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_classes: int, number of classes.
        """
        super(LinearClassifier, self).__init__(in_channels, num_classes, dropout)

    @property
    def num_classes(self):
        return self.num_features


class SHAPClassifier(GAPHeadBase):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output features.
        """
        super(SHAPClassifier, self).__init__(in_channels, num_classes)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(in_channels: int, num_classes: int, dropout: float = 0.0):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('dropout', nn.Dropout(p=dropout)),
                    ('linear', nn.Linear(in_channels, num_classes))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        out = nn.AdaptiveAvgPool2d(1)(x)
        out = out.view(out.size(0), -1)
        out = self.layers(out)
        return out

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)