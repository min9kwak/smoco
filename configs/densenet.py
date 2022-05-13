import argparse
from configs.base import ConfigBase


class DenseNetMRIConfig(ConfigBase):
    """Configurations for MoCo."""

    def __init__(self, args=None, **kwargs):
        super(DenseNetMRIConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Unimodal DenseNet', add_help=False)
        parser.add_argument('--init_features', type=int, default=64)
        parser.add_argument('--growth_rate', type=int, default=32)
        parser.add_argument('--block_config', type=str, default='6,12,24,16')
        parser.add_argument('--bn_size', type=int, default=4)
        parser.add_argument('--dropout_rate', type=float, default=0.0)
        return parser

    @property
    def task(self) -> str:
        return 'mri'

    @property
    def backbone(self) -> str:
        return 'densenet'
