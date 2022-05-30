# TODO: check loading speed of MRI (mgz and pkl)
import pickle
import os
import nibabel as nib

DATA_DIR = "/raidWorkspace/mingu/Data/ADNI/FS7/m127S0925L111010M615TCF/mri"
image_file = os.path.join(DATA_DIR, 'brain.mgz')

import time

s1 = time.time()
image = nib.load(image_file)
image = nib.as_closest_canonical(image)
image = image.get_fdata()
e1 = time.time()
print('mri loading mgz', e1 - s1)

s2 = time.time()
with open(image_file.replace('.mgz', '.pkl'), 'rb') as f:
    image = pickle.load(f)
e2 = time.time()
print('mri loading pkl', e2 - s2)


nii_file = "/raidWorkspace/mingu/Data/ADNI/PUP_FBP\FFBP002S0729L073010M4/pet_proc/FBP002S0729L073010M4_SUVR.nii"
pkl_file = "/raidWorkspace/mingu/Data/ADNI/PUP_FBP\FFBP002S0729L073010M4/pet_proc/FBP002S0729L073010M4_SUVR.pkl"
mask_file = "/raidWorkspace/mingu/Data/ADNI/FS7/m002S0729L072210M715TCF/mri/brainmask.mgz"

s1 = time.time()
mask, image = nib.load(mask_file), nib.load(nii_file)
mask, image = nib.as_closest_canonical(mask), nib.as_closest_canonical(image)
mask, image = mask.get_fdata(), image.get_fdata().squeeze()
mask = (mask == 0)
image[mask] = 0
e1 = time.time()
print('pet loading nii', e1 - s1)

s2 = time.time()
with open(pkl_file, 'rb') as f:
    image = pickle.load(f)
e2 = time.time()
print('pet loading pkl', e2 - s2)
