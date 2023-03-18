import os

STR2GPU = {
    '00': 'MIG-f3304720-4601-5894-bee4-cd0174024e06',
    '10': 'MIG-35d812cd-2b57-5e55-bb30-890bd9675846',
    '20': 'MIG-2537493a-e4ca-5b97-850f-4c9e28fa2863',
    '21': 'MIG-3feaa31a-e279-5752-a99e-573825c15aac',
    '22': 'MIG-b62af7d4-3a7b-5352-9a6b-e675d109543f',
    '23': 'MIG-e7905fdd-bb2f-5c66-8b44-0ecca5228ed8',
    '24': 'MIG-8e436ac2-282e-5f88-810b-a9e2812fa7ba',
    '25': 'MIG-a4978042-26f3-5231-be6a-e991b7d95832',
    '26': 'MIG-5e1cfae4-7158-5c8a-ae42-30be508d82b5',
    '30': 'MIG-cb7e2c1e-63d3-507d-a862-1e574a43ea4f',
    '31': 'MIG-1c6e06b0-0504-5626-ad12-df578a2e724b',
    '32': 'MIG-9dc4e79d-857a-53cb-a308-2cf6bdc5ef96',
    '33': 'MIG-2d7c91c7-e4ee-5584-9c8a-bf6a090d7b96',
    '34': 'MIG-5c5c61db-3a8e-5e53-809e-d9d06a8e8c73',
    '35': 'MIG-4baea7de-6e55-5af9-a023-c238d32148b2',
    '36': 'MIG-7cd7815e-16cd-5648-b3a3-b4f7bd1f1f0a'
}


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
