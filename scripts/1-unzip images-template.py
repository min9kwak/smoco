import os
import glob
import tarfile
import gzip

import pandas as pd
import tqdm
import shutil
import pickle

# define directory
MRI_DIR_FROM = "D:/data/ADNI/source files/FS7"
MRI_DIR_TO = "D:/data/ADNI/FS7/"

FBP_DIR_FROM = "D:/data/ADNI/source files/PUP_FBP/"
FBP_DIR_TO = "D:/data/ADNI/PUP_FBP/"

os.makedirs(MRI_DIR_TO, exist_ok=True)
os.makedirs(FBP_DIR_TO, exist_ok=True)

# MRI
mri_parents = glob.glob(os.path.join(MRI_DIR_FROM, "*.tar"))
logs = []
for mri_parent in tqdm.tqdm(mri_parents):

    tar_parent = tarfile.TarFile(mri_parent)
    tar_parent_names = tar_parent.getnames()

    path_parent = mri_parent.replace(".tar", "")
    tar_parent.extractall(path=path_parent)

    for tar_parent_name in tqdm.tqdm(tar_parent_names):

        tar_child = tarfile.open(os.path.join(path_parent, tar_parent_name))
        tar_child_names = tar_child.getnames()
        wmm_names = [n for n in tar_child_names if 'cat12/mri/wmm' in n]
        tar_child_names = wmm_names
        for tar_child_name in tar_child_names:
            tar_child.extract(tar_child_name, path=os.path.join(MRI_DIR_TO))
        del tar_child
    shutil.rmtree(path_parent, ignore_errors=True)




mri_filenames = glob.glob(os.path.join('D:/data/ADNI/template/FS7', '*/cat12/mri/wmm*.nii.gz'), recursive=True)
mri_names = [m.split('\\')[1] for m in mri_filenames]

pet_filenames = glob.glob(os.path.join('D:/data/ADNI/PUP_FBP', "*/pet_proc/w_*_SUVR.nii.gz"), recursive=True)
pet_names = [p.split('\\')[1] for p in pet_filenames]

import pandas as pd
data_info = pd.read_csv(os.path.join('D:/data/ADNI/labels/data_info.csv'))

for pet_filename in pet_filenames:
    pet_name = pet_filename.split('\\')[1]
    if data_info.loc[data_info.PET == pet_name].MRI.item() in mri_names:
        pet = nib.load(pet_filename).get_fdata()

        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs = axs.ravel()
        sns.heatmap(pet[60, :, :], vmin=0, vmax=3.4, ax=axs[0], cmap='binary')
        sns.heatmap(pet[:, 70, :], vmin=0, vmax=3.4, ax=axs[1], cmap='binary')
        sns.heatmap(pet[:, :, 60], vmin=0, vmax=3.4, ax=axs[2], cmap='binary')
        plt.tight_layout()
        plt.show()

        mri_name = data_info.loc[data_info.PET == pet_name].MRI.item()
        mri_filename = [f for f in mri_filenames if mri_name in f][0]
        mri = nib.load(mri_filename).get_fdata()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs = axs.ravel()
        sns.heatmap(mri[60, :, :], vmin=0, vmax=1.15, ax=axs[0], cmap='binary')
        sns.heatmap(mri[:, 70, :], vmin=0, vmax=1.15, ax=axs[1], cmap='binary')
        sns.heatmap(mri[:, :, 60], vmin=0, vmax=1.15, ax=axs[2], cmap='binary')
        plt.tight_layout()
        plt.show()

        mask = (mri == 0)
        pet_brain = pet.copy()
        pet_brain[mask] = 0

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs = axs.ravel()
        sns.heatmap(pet_brain[60, :, :], vmin=0, vmax=3.4, ax=axs[0], cmap='binary')
        sns.heatmap(pet_brain[:, 70, :], vmin=0, vmax=3.4, ax=axs[1], cmap='binary')
        sns.heatmap(pet_brain[:, :, 60], vmin=0, vmax=3.4, ax=axs[2], cmap='binary')
        plt.tight_layout()
        plt.show()


        pet_brain.shape
        import numpy as np

        padded = np.pad(pet_brain, ((12, 12), (0, 0), (12, 12)), 'constant')
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs = axs.ravel()
        sns.heatmap(padded[70, :, :], vmin=0, vmax=3.4, ax=axs[0], cmap='binary')
        sns.heatmap(padded[:, 70, :], vmin=0, vmax=3.4, ax=axs[1], cmap='binary')
        sns.heatmap(padded[:, :, 70], vmin=0, vmax=3.4, ax=axs[2], cmap='binary')
        plt.tight_layout()
        plt.show()
