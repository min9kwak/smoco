import argparse
from configs.base import ConfigBase


class VitUniConfig(ConfigBase):
    """Configurations for MoCo."""

    def __init__(self, args=None, **kwargs):
        super(VitUniConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Unimodal Vit', add_help=False)
        parser.add_argument('--patch_size', type=int, default=24, choices=(24, 16))
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--mlp_dim', type=int, default=2048)
        parser.add_argument('--num_layers', type=int, default=8)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--pos_embed', type=str, default='conv', choices=('conv', 'perceptron'))
        parser.add_argument('--dropout_rate', type=float, default=0.0)
        return parser

    @property
    def backbone(self) -> str:
        return 'vit'
