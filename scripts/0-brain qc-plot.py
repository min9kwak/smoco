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

plt_dir = 'C:/Users/mingu/Desktop/MRI QC PLOT'
os.makedirs(plt_dir, exist_ok=True)

import matplotlib.pyplot as plt
import seaborn as sns

for brain_file in tqdm.tqdm(brain_files):

    id = brain_file.split('\\')[1]

    brain = nib.load(brain_file)
    brain = nib.as_closest_canonical(brain)
    brain = brain.get_fdata()

    vmax = brain.max()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs = axs.ravel()
    sns.heatmap(brain[128, :, :], vmin=0, vmax=vmax, ax=axs[0])
    sns.heatmap(brain[:, 128, :], vmin=0, vmax=vmax, ax=axs[1])
    sns.heatmap(brain[:, :, 128], vmin=0, vmax=vmax, ax=axs[2])
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, f'{id}.png'),
                bbox_inches='tight')
    plt.close()

#####
id = 'm029S1384L032807M515TCF'
brain_file = os.path.join(DATA_DIR, id, 'mri/brain.mgz')

brain = nib.load(brain_file)
brain = nib.as_closest_canonical(brain)
brain = brain.get_fdata()

vmax = brain.max()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs = axs.ravel()
sns.heatmap(brain[160, :, :], vmin=0, vmax=vmax, ax=axs[0])
sns.heatmap(brain[:, 128, :], vmin=0, vmax=vmax, ax=axs[1])
sns.heatmap(brain[:, :, 128], vmin=0, vmax=vmax, ax=axs[2])
plt.tight_layout()
plt.show()



#############
DATA_DIR = "D:/data/ADNI/PUP_FBP"
data_info = pd.read_csv('D:/data/ADNI/labels/data_info.csv')
data_info = data_info[~data_info.PET.isna()]
pet_files = sorted([os.path.join(DATA_DIR, row.PET, f'pet_proc/{row.PET}_SUVR.pkl') for _, row in data_info.iterrows()])

plt_dir = 'C:/Users/mingu/Desktop/PET QC PLOT'
os.makedirs(plt_dir, exist_ok=True)
import pickle

for pet_file in tqdm.tqdm(pet_files):

    id = pet_file.split('\\')[1]

    with open(pet_file, 'rb') as fb:
        pet = pickle.load(fb)

    vmax = pet.max()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs = axs.ravel()
    sns.heatmap(pet[128, :, :], vmin=0, vmax=vmax, ax=axs[0])
    sns.heatmap(pet[:, 128, :], vmin=0, vmax=vmax, ax=axs[1])
    sns.heatmap(pet[:, :, 128], vmin=0, vmax=vmax, ax=axs[2])
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, f'{id}.png'),
                bbox_inches='tight')
    plt.close()
