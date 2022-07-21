import argparse
from configs.base import ConfigBase


class MixUpConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(MixUpConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Classification', add_help=False)
        parser.add_argument('--semi', action='store_true')
        parser.add_argument('--balance', action='store_true', help='apply class balance weight')
        parser.add_argument('--alpha', type=float, default=0.3, help='hyperparameter of beta distribution')
        return parser
