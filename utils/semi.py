import torch
import torch.nn as nn

class SemiBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, channels):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )


def update_batch_stats(module, flag):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            child.update_batch_stats = flag
    for n, ch in module.named_children():
        update_batch_stats(ch, flag)


def replace_bn(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm3d:
            print('replaced: ', name, attr_str)
            setattr(module, attr_str, SemiBatchNorm3d(target_attr.num_features))
    for n, ch in module.named_children():
        replace_bn(ch, n)
