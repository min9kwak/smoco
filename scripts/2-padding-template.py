import os
import tqdm
import numpy as np
import pandas as pd
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

root = '/raidWorkspace/mingu/Data/ADNI'
data_info = pd.read_csv(os.path.join(root, 'labels/data_info.csv'))
mri_save_dir = os.path.join(root, "template/FS7")
pet_save_dir = os.path.join(root, "template/PUP_FBP")
os.makedirs(mri_save_dir, exist_ok=True)
os.makedirs(pet_save_dir, exist_ok=True)


def calculate_crop_size(m, M, l):
    hl = l // 2
    center = m + (M - m) // 2
    tm = center - hl
    tM = center + hl
    if tm < 0:
        tM = tM - tm
        tm = 0
    if tM > 255:
        tm = tm + 255 - tM
        tM = 255
    return tm, tM


def slice_image(image, lengths):
    points = []
    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(image, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        tm, tM = calculate_crop_size(m, M, lengths[i])
        points.extend([tm, tM])
    image = image[points[0]:points[1], points[2]:points[3], points[4]:points[5]]
    return image

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
    mri_pad = slice_image(mri_pad, [130, 130, 130])
    with open(os.path.join(mri_save_dir, f'{row.MRI}.pkl'), 'wb') as fb:
        pickle.dump(mri_pad, fb)

    if pet_file is not None:
        pet = nib.load(pet_file).get_fdata()
        pet = np.nan_to_num(pet)
        mask = (mri == 0)
        pet[mask] = 0
        pet_pad = np.pad(pet, ((12, 12), (0, 0), (12, 12)), 'constant')
        pet_pad = slice_image(pet_pad, [130, 130, 130])
        with open(os.path.join(pet_save_dir, f'{row.PET}.pkl'), 'wb') as fb:
            pickle.dump(mri_pad, fb)
