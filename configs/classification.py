import argparse
from configs.base import ConfigBase


class ClassificationConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(ClassificationConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Classification', add_help=False)
        parser.add_argument('--semi', action='store_true')

        return parser
