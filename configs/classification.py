import argparse
from configs.base import ConfigBase


class ClassificationConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(ClassificationConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Classification', add_help=False)
        parser.add_argument('--semi', action='store_true')
        parser.add_argument('--balance', action='store_true', help='apply class balance weight')

        return parser


class SWAConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(SWAConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Classification', add_help=False)
        parser.add_argument('--semi', action='store_true')
        parser.add_argument('--semi_loss', type=str, choices=('pi', 'pseudo'))
        parser.add_argument('--pseudo_threshold', type=float)
        parser.add_argument('--balance', action='store_true', help='apply class balance weight')
        parser.add_argument('--swa_learning_rate', type=float, default=1e-4)
        parser.add_argument('--swa_start', type=int, default=0)
        parser.add_argument('--mu', type=int, default=3, help='multiplier for unlabeled batch size.')
        parser.add_argument('--alpha', type=float, default=100.0, help='coefficient for unlabeled loss.')
        parser.add_argument('--ramp_up', type=int, default=5, help='ramp up for alpha.')

        return parser
