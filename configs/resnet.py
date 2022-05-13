import argparse
from configs.base import ConfigBase


class ResNetMRIConfig(ConfigBase):
    """Configurations for MoCo."""

    def __init__(self, args=None, **kwargs):
        super(ResNetMRIConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Unimodal DenseNet', add_help=False)
        parser.add_argument('--arch', type=int, default=50)
        parser.add_argument('--no_max_pool', action='store_true')
        return parser

    @property
    def task(self) -> str:
        return 'mri'

    @property
    def backbone(self) -> str:
        return 'resnet'

# setattr(config, 'target_label', [int(a) for a in config.target_label.split(',')])
