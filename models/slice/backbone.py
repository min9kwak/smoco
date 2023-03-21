# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet101, wide_resnet50_2
from utils.initialization import initialize_weights


RESNET_FUNCTIONS = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'wide50_2': wide_resnet50_2
}


class BackboneBase(nn.Module):
    def __init__(self, in_channels: int):
        super(BackboneBase, self).__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.FloatTensor):
        raise NotImplementedError

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False

    def save_weights_to_checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights_from_checkpoint(self, path: str, key: str):
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetBackbone(BackboneBase):
    def __init__(self, name: str = 'resnet50', in_channels: int = 3):
        super(ResNetBackbone, self).__init__(in_channels)

        self.name = name  # resnet18, resnet50, resnet101
        try:
            self.layers = RESNET_FUNCTIONS[self.name](weights=None)
        except:
            self.layers = RESNET_FUNCTIONS[self.name](pretrained=False)

        self.layers = self._remove_gap_and_fc(self.layers)
        if self.in_channels != 3:
            self.layers = self._fix_first_conv_in_channels(self.layers, in_channels=self.in_channels)
        initialize_weights(self)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

    @staticmethod
    def _fix_first_conv_in_channels(resnet: nn.Module, in_channels: int) -> nn.Module:
        """
        Change the number of incoming channels for the first layer.
        """
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'conv1':
                conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                model.add_module(name, conv1)
            else:
                model.add_module(name, child)

        return model

    @staticmethod
    def _remove_gap_and_fc(resnet: nn.Module) -> nn.Module:
        """
        Remove global average pooling & fully-connected layer
        For torchvision ResNet models only."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)  # preserve original names

        return model

    def _fix_first_conv(self):
        conv1 = self.layers.conv1
        self.layers.conv1 = nn.Conv2d(conv1.in_channels, conv1.out_channels,
                                      kernel_size=3, stride=1, padding=1, bias=False)

    @staticmethod
    def _remove_maxpool(resnet: nn.Module):
        """Remove first max pooling layer of ResNet."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'maxpool':
                continue
            else:
                model.add_module(name, child)

        return model

    @property
    def out_channels(self):
        if self.name == 'resnet18':
            return 512
        else:
            return 2048


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bn_momentum=0.1, leaky_slope=0.0, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, bn_momentum=0.1, leaky_slope=0.0, dropout=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, bn_momentum, leaky_slope,
                                      dropout)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, bn_momentum, leaky_slope, dropout):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, bn_momentum, leaky_slope, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideBackbone(BackboneBase):
    def __init__(self,
                 in_channels: int = 3, depth: int = 28, widen_factor: int = 2,
                 bn_momentum: float = 0.1, leaky_slope: float = 0.1, dropout: float = 0.0
                 ):
        super(WideBackbone, self).__init__(in_channels)

        self.head = None

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # blocks
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, bn_momentum, leaky_slope, dropout)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, bn_momentum, leaky_slope, dropout)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, bn_momentum, leaky_slope, dropout)

        # bn and activation
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=bn_momentum)
        self.relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.nChannels = nChannels[3]

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        if self.head:
            out = self.head(out)
        return out

    def add_head(self, head: nn.Module):
        self.head = head

    @property
    def out_channels(self):
        return 128


if __name__ == '__main__':

    import torch
    batch = torch.rand(size=(10, 1, 64, 64)).cuda()
    backbone = ResNetBackbone(name='resnet50', in_channels=1)
    backbone.cuda()
    h = backbone(batch)
    print(h.shape)
    print(backbone.layers.conv1)

    backbone._fix_first_conv()
    backbone.cuda()
    h = backbone(batch)
    print(h.shape)
    print(backbone.layers.conv1)

    from models.slice.head import LinearClassifier
    classifier = LinearClassifier(in_channels=backbone.out_channels, num_classes=2)
    classifier.cuda()
    logit = classifier(h)
    print(logit.shape)