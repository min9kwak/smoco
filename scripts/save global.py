import os
import pickle
import tqdm
import nibabel as nib
import pandas as pd
import numpy as np


# PET - left hippocampus
root = 'D:/data/ADNI'
data_info = 'labels/data_info.csv'

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


data_info = pd.read_csv(os.path.join(root, data_info),
                        converters={'RID': str, 'MONTH': int, 'Conv': int})
with open(os.path.join(root, 'labels/mri_abnormal.pkl'), 'rb') as fb:
    mri_abnormal = pickle.load(fb)
data_info = data_info.loc[~data_info.MRI.isin(mri_abnormal)]

os.makedirs(os.path.join(root, 'segment', 'FS7', 'global'), exist_ok=True)
os.makedirs(os.path.join(root, 'segment', 'PUP_FBP', 'global'), exist_ok=True)


for _, row in tqdm.tqdm(data_info.iterrows(), total=len(data_info)):

    brainmask_file = os.path.join(root, 'FS7', row.MRI, 'mri/brainmask.mgz')
    brain_file = os.path.join(root, 'FS7', row.MRI, 'mri/brain.mgz')
    if type(row.PET) == str:
        pet_file = os.path.join(root, 'PUP_FBP', row.PET, f'pet_proc/{row.PET}_SUVR.pkl')
    else:
        pet_file = None

    brain = nib.load(brain_file)
    brain = nib.as_closest_canonical(brain)
    brain = brain.get_fdata()

    # save mri
    brain = slice_image(brain, [196, 196, 196])
    save_name = os.path.join(root, 'segment', 'FS7', 'global', f'{row.MRI}.pkl')
    with open(save_name, 'wb') as f:
        pickle.dump(brain, f)

    if pet_file is not None:

        brainmask = nib.load(brainmask_file)
        brainmask = nib.as_closest_canonical(brainmask)
        brainmask = brainmask.get_fdata()

        with open(pet_file, 'rb') as f:
            pet = pickle.load(f)
        pet_brain = pet * (brainmask != 0)
        pet_brain = slice_image(pet_brain, [196, 196, 196])
        save_name = os.path.join(root, 'segment', 'PUP_FBP', 'global', f'{row.PET}.pkl')
        with open(save_name, 'wb') as f:
            pickle.dump(pet_brain, f)
