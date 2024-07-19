import os
import tqdm
import numpy as np
import pandas as pd
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "D:/data/ADNI/FS7"
data_info = pd.read_csv('D:/data/ADNI/labels/data_info.csv')
brain_files = sorted([os.path.join(DATA_DIR, f, 'mri/brain.mgz') for f in data_info.MRI.tolist()])

stats = []
for brain_file in tqdm.tqdm(brain_files):

    id = brain_file.split('\\')[1]

    brain = nib.load(brain_file)
    brain = nib.as_closest_canonical(brain)
    brain = brain.get_fdata()

    # background (zero) ratio
    background_ratio = np.sum(brain == 0) / np.prod(brain.shape)

    # foreground voxel statistics
    brain_fore = brain[brain != 0]
    brain_fore_mean = np.mean(brain_fore)
    brain_fore_median = np.median(brain_fore)
    brain_fore_std = np.std(brain_fore)

    # brain-area size - dimension
    brain_size = []
    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(brain, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        length = M - m
        brain_size.append(length)

    stat = [id, background_ratio,
            brain_fore_mean, brain_fore_median, brain_fore_std,
            *brain_size]
    stats.append(stat)

import pickle
with open('mri_stats.pkl', 'wb') as f:
    pickle.dump(stats, f)

stats = pd.DataFrame(stats)
stats.columns = ['id', 'back_ratio', 'fore_mean', 'fore_median', 'fore_std', 'len_x', 'len_y', 'len_z']

import matplotlib.pyplot as plt
candidates = []
for column in stats.columns[1:]:
    values = stats[column].values
    plt.hist(values, bins=500)
    plt.show()
    c1 = list(np.where(values < np.percentile(values, 0.1))[0])
    c2 = list(np.where(values > np.percentile(values, 99.9))[0])
    candidates.extend(c1 + c2)
candidates = np.unique(candidates)
stats.iloc[candidates, :]