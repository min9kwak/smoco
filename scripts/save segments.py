import os
import pickle
import tqdm
import nibabel as nib
import pandas as pd
import numpy as np


# PET - left hippocampus
root = 'D:/data/ADNI'
data_info = 'labels/data_info.csv'

segment2class = {'left_hippocampus': [17],
                 'right_hippocampus': [53],
                 'hippocampus': [17, 53]}
LENGTH = {'left_hippocampus': [36, 64, 64],
          'right_hippocampus': [64, 96, 64],
          'hippocampus': [96, 96, 96]}


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

for segment in segment2class.keys():

    dir1 = os.path.join(root, 'segment', 'FS7', segment)
    dir2 = os.path.join(root, 'segment', 'PUP_FBP', segment)
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)

for _, row in tqdm.tqdm(data_info.iterrows(), total=len(data_info)):

    wmparc_file = os.path.join(root, 'FS7', row.MRI, 'mri/wmparc.mgz')
    brain_file = os.path.join(root, 'FS7', row.MRI, 'mri/brain.mgz')
    if type(row.PET) == str:
        pet_file = os.path.join(root, 'PUP_FBP', row.PET, f'pet_proc/{row.PET}_SUVR.pkl')
    else:
        pet_file = None

    wmparc = nib.load(wmparc_file)
    wmparc = nib.as_closest_canonical(wmparc)
    wmparc = wmparc.get_fdata()

    brain = nib.load(brain_file)
    brain = nib.as_closest_canonical(brain)
    brain = brain.get_fdata()

    # hippocampus for mri
    for segment in segment2class.keys():
        hippo = brain * np.isin(wmparc, segment2class[segment])
        hippo = slice_image(hippo, LENGTH[segment])
        save_name = os.path.join(root, 'segment', 'FS7', segment, f'{row.MRI}.pkl')
        with open(save_name, 'wb') as f:
            pickle.dump(hippo, f)

    if pet_file is not None:
        with open(pet_file, 'rb') as f:
            pet = pickle.load(f)
        for segment in segment2class.keys():
            hippo = pet * np.isin(wmparc, segment2class[segment])
            hippo = slice_image(hippo, LENGTH[segment])
            save_name = os.path.join(root, 'segment', 'PUP_FBP', segment, f'{row.PET}.pkl')
            with open(save_name, 'wb') as f:
                pickle.dump(hippo, f)
