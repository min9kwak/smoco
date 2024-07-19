# -*- coding: utf-8 -*-

import argparse
from configs.base import ConfigBase, str2bool


class AIBLConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(ConfigBase, self).__init__(args, **kwargs)

    @property
    def finetune_type(self) -> str:
        return 'finetune'


    @staticmethod
    def finetune_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Finetune", add_help=False)
        parser.add_argument('--pretrained_dir', type=str, default=None, help='Path to pretrained model file (.pt).')
        parser.add_argument('--freeze_bn', type=str2bool, default=False, help='Freeze BN weights of CNN backbone.')
        parser.add_argument('--balance', type=str2bool, default=True, help='apply class balance weight')
        parser.add_argument('--finetune_trans', type=str, default='test', choices=('train', 'test'))

        # data
        parser.add_argument('--root', type=str, default='/raidWorkspace/mingu/Data/AIBL')
        parser.add_argument('--data_info', type=str, default='data_info.csv')
        parser.add_argument('--time_window', type=int, default=36, choices=(18, 36))
        parser.add_argument('--random_state', type=int, default=2021)
        parser.add_argument('--n_splits', type=int, default=5)
        parser.add_argument('--n_cv', type=int, default=0)

        parser.add_argument('--train_mode', type=str, default='train', choices=('train', 'test'))
        return parser
