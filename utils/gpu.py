import os

STR2GPU = {'00': 'MIG-4e9bdbba-d0ea-5377-ae8a-a78ccab2f5e5',
           '01': 'MIG-f45e64c7-dc06-5453-a81e-9bc9ecc30588',
           '10': 'MIG-91fc8fce-9c3d-57d0-b652-6270d2d1d7d4',
           '11': 'MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf',
           '20': 'MIG-dc45e153-fb1e-5b2d-8a31-d3fb9494cd80',
           '21': 'MIG-21d343f4-de6e-5d44-9774-e2f3dbab968d',
           '30': 'MIG-21d343f4-de6e-5d44-9774-e2f3dbab968d',
           '31': 'MIG-21d343f4-de6e-5d44-9774-e2f3dbab968d'}


def set_gpu(config):
    if config.server == 'workstation2':
        gpus = ','.join([STR2GPU[str(gpu)] for gpu in config.gpus])
    else:
        gpus = ','.join([str(gpu) for gpu in config.gpus])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
    parser.add_argument('--gpus', type=str, nargs='+', default=None, help='')
    parser.add_argument('--server', type=str)

    config = parser.parse_args()
    set_gpu(config)
