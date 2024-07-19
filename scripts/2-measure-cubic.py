import os
import tqdm
import numpy as np
import pandas as pd
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

DATA_DIR = "D:/data/ADNI"
data_info = pd.read_csv('D:/data/ADNI/labels/data_info.csv')
with open(os.path.join(DATA_DIR, 'labels/mri_abnormal.pkl'), 'rb') as fb:
    mri_abnormal = pickle.load(fb)
data_info = data_info.loc[~data_info.MRI.isin(mri_abnormal)]

# MRI brain - whole
brain_files = sorted([os.path.join(DATA_DIR, 'FS7', row.MRI, 'mri/brain.mgz') for _, row in data_info.iterrows()])
brain_dims = {k: {'min': 255, 'max': 0, 'len': 0} for k in range(3)}
for brain_file in tqdm.tqdm(brain_files):

    # check ids are matched
    brain = nib.load(brain_file)
    brain = nib.as_closest_canonical(brain)
    brain = brain.get_fdata()

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

# MRI - left and right hippocampus
wmparc_files = sorted([os.path.join(DATA_DIR, 'FS7', row.MRI, 'mri/wmparc.mgz') for _, row in data_info.iterrows()])
left_hippo_dims = {k: {'min': 255, 'max': 0, 'len': 0} for k in range(3)}
right_hippo_dims = {k: {'min': 255, 'max': 0, 'len': 0} for k in range(3)}
hippo_dims = {k: {'min': 255, 'max': 0, 'len': 0} for k in range(3)}

for wmparc_file in tqdm.tqdm(wmparc_files):
    wmparc = nib.load(wmparc_file)
    wmparc = nib.as_closest_canonical(wmparc)
    wmparc = wmparc.get_fdata()

    left_hippo = (wmparc == 17)
    right_hippo = (wmparc == 53)

    hippo = left_hippo + right_hippo

    left_hippo = np.array(left_hippo, dtype=float)
    right_hippo = np.array(right_hippo, dtype=float)
    hippo = np.array(hippo, dtype=float)

    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(left_hippo, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        l = M - m
        if m < left_hippo_dims[i]['min']:
            left_hippo_dims[i]['min'] = m
        if M > left_hippo_dims[i]['max']:
            left_hippo_dims[i]['max'] = M
        if l > left_hippo_dims[i]['len']:
            left_hippo_dims[i]['len'] = l
        del s, mask, m, M, l

    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(right_hippo, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        l = M - m
        if m < right_hippo_dims[i]['min']:
            right_hippo_dims[i]['min'] = m
        if M > right_hippo_dims[i]['max']:
            right_hippo_dims[i]['max'] = M
        if l > right_hippo_dims[i]['len']:
            right_hippo_dims[i]['len'] = l
        del s, mask, m, M, l

    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(hippo, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        l = M - m
        if m < hippo_dims[i]['min']:
            hippo_dims[i]['min'] = m
        if M > hippo_dims[i]['max']:
            hippo_dims[i]['max'] = M
        if l > hippo_dims[i]['len']:
            hippo_dims[i]['len'] = l
        del s, mask, m, M, l

