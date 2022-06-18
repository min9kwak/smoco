import os
import tqdm
import numpy as np
import pandas as pd
import glob
import pickle
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "D:/data/ADNI/TEMPLATE/FS7"
data_info = pd.read_csv('D:/data/ADNI/labels/data_info.csv')
brain_files = sorted(glob.glob(os.path.join(DATA_DIR, "**.pkl")))
brain_dims = {k: {'min': 255, 'max': 0, 'len': 0} for k in range(3)}
for brain_file in tqdm.tqdm(brain_files):
    with open(brain_file, 'rb') as fb:
        brain = pickle.load(fb)
    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(brain, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        l = M - m
        if m < brain_dims[i]['min']:
            brain_dims[i]['min'] = m
        if M > brain_dims[i]['max']:
            brain_dims[i]['max'] = M
        if l > brain_dims[i]['len']:
            brain_dims[i]['len'] = l

