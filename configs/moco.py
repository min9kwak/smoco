import argparse
from configs.base import ConfigBase


class MoCoConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(MoCoConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('MoCO', add_help=False)
        parser.add_argument('--projector_dim', type=int, default=128, help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.2, help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=512, help='Number of negative examples to maintain.')
        parser.add_argument('--key_momentum', type=float, default=0.999, help='Momentum for updating key encoder.')
        parser.add_argument('--split_bn', action='store_true')
        parser.add_argument('--knn_k', type=str, default="1, 5, 15", help='')
        return parser
