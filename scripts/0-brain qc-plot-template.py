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

plt_dir = 'C:/Users/mingu/Desktop/MRI TEMPLATE QC PLOT'
os.makedirs(plt_dir, exist_ok=True)


for brain_file in tqdm.tqdm(brain_files):

    id = brain_file.split('\\')[1].replace('.pkl', '')
    with open(brain_file, 'rb') as fb:
        brain = pickle.load(fb)

    vmax = brain.max()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs = axs.ravel()
    sns.heatmap(brain[70, :, :], vmin=0, vmax=vmax, ax=axs[0], cmap='binary')
    sns.heatmap(brain[:, 70, :], vmin=0, vmax=vmax, ax=axs[1], cmap='binary')
    sns.heatmap(brain[:, :, 70], vmin=0, vmax=vmax, ax=axs[2], cmap='binary')
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, f'{id}.png'),
                bbox_inches='tight')
    plt.close()


plt_dir = 'C:/Users/mingu/Desktop/PET TEMPLATE QC PLOT'
pet_files = sorted(glob.glob(os.path.join("D:/data/ADNI/TEMPLATE/PUP_FBP", '**.pkl')))
for pet_file in tqdm.tqdm(pet_files):
    id = pet_file.split('\\')[1].replace('.pkl', '')
    with open(pet_file, 'rb') as fb:
        brain = pickle.load(fb)

    vmax = brain.max()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs = axs.ravel()
    sns.heatmap(brain[70, :, :], vmin=0, vmax=vmax, ax=axs[0], cmap='binary')
    sns.heatmap(brain[:, 70, :], vmin=0, vmax=vmax, ax=axs[1], cmap='binary')
    sns.heatmap(brain[:, :, 70], vmin=0, vmax=vmax, ax=axs[2], cmap='binary')
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, f'{id}.png'),
                bbox_inches='tight')
    plt.close()
