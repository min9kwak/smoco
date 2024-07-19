# -*- coding: utf-8 -*-

import os
import copy
import json
import argparse
import datetime
from configs.base import ConfigBase, str2bool


class FinetuneConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(ConfigBase, self).__init__(args, **kwargs)

    @property
    def finetune_type(self) -> str:
        if self.freeze:
            return 'linear'
        else:
            return 'finetune'

    @staticmethod
    def finetune_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Finetune", add_help=False)
        parser.add_argument('--pretrained_dir', type=str, default=None, help='Path to pretrained model file (.pt).')
        parser.add_argument('--freeze', action='store_true', help='Freeze weights of CNN backbone.')
        parser.add_argument('--freeze_bn', action='store_true', help='Freeze BN weights of CNN backbone.')
        parser.add_argument('--balance', action='store_true', help='apply class balance weight')
        parser.add_argument('--finetune_trans', type=str, default='test', choices=('train', 'test'))
        return parser


class DemoFinetuneConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(ConfigBase, self).__init__(args, **kwargs)

    @property
    def finetune_type(self) -> str:
        if self.freeze:
            return 'linear'
        else:
            return 'finetune'

    @staticmethod
    def finetune_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Finetune", add_help=False)
        parser.add_argument('--pretrained_dir', type=str, default=None, help='Path to pretrained model file (.pt).')
        parser.add_argument('--freeze', action='store_true', help='Freeze weights of CNN backbone.')
        parser.add_argument('--freeze_bn', action='store_true', help='Freeze BN weights of CNN backbone.')
        parser.add_argument('--balance', action='store_true', help='apply class balance weight')
        parser.add_argument('--finetune_trans', type=str, default='test', choices=('train', 'test'))

        parser.add_argument('--hidden', type=str, default="3")
        parser.add_argument('--add_apoe', action='store_true')
        parser.add_argument('--add_volume', action='store_true')

        return parser
