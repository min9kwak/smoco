import os
import tqdm
import numpy as np
import pandas as pd
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

root = 'D:/data/ADNI'
# root = '/raidWorkspace/mingu/Data/ADNI'
data_info = pd.read_csv(os.path.join(root, 'labels/data_info.csv'))
mri_save_dir = os.path.join(root, "template1/FS7")
pet_save_dir = os.path.join(root, "template1/PUP_FBP")
os.makedirs(mri_save_dir, exist_ok=True)
os.makedirs(pet_save_dir, exist_ok=True)

# filter
data_info = data_info.loc[data_info.IS_FILE]
brain_files = [os.path.join(root, 'FS7', row.MRI, f'cat12/mri/wm{row.MRI}N.nii.gz')
               for _, row in data_info.iterrows()]

for _, row in tqdm.tqdm(data_info.iterrows(), total=len(data_info)):

    mri_file = os.path.join(root, 'FS7', row.MRI, f'cat12/mri/wm{row.MRI}N.nii.gz')
    if type(row.PET) == str:
        pet_file = os.path.join(root, 'PUP_FBP', row.PET, f'pet_proc/w_{row.PET}_SUVR.nii.gz')
    else:
        pet_file = None

    mri = nib.load(mri_file).get_fdata()
    mri_pad = np.pad(mri, ((12, 12), (0, 0), (12, 12)), 'constant')
    with open(os.path.join(mri_save_dir, f'{row.MRI}.pkl'), 'wb') as fb:
        pickle.dump(mri_pad, fb)

    if pet_file is not None:
        pet = nib.load(pet_file).get_fdata()
        pet = np.nan_to_num(pet)
        mask = (mri == 0)
        pet[mask] = 0
        pet_pad = np.pad(pet, ((12, 12), (0, 0), (12, 12)), 'constant')
        with open(os.path.join(pet_save_dir, f'{row.PET}.pkl'), 'wb') as fb:
            pickle.dump(pet_pad, fb)
