import argparse
from configs.base import ConfigBase


class SupMoCoConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(SupMoCoConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('MoCO', add_help=False)
        parser.add_argument('--projector_dim', type=int, default=128, help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.2, help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=512, help='Number of negative examples to maintain.')
        parser.add_argument('--key_momentum', type=float, default=0.999, help='Momentum for updating key encoder.')
        parser.add_argument('--split_bn', action='store_true')
        parser.add_argument('--knn_k', type=str, default="1, 5, 15", help='')
        parser.add_argument('--alphas', type=str, default="1.0, 1.0", help='weights for losses')
        parser.add_argument('--alphas_min', type=str, default="1.0, 0.0",
                            help='minimum values of weights for losses')
        parser.add_argument('--alphas_decay_end', type=str, default="-1, -1")
        return parser
