# TODO: check loading speed of MRI (mgz and pkl)
import glob
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
print(e1 - s1)

s2 = time.time()
with open(image_file.replace('.mgz', '.pkl'), 'rb') as f:
    image = pickle.load(f)
e2 = time.time()
print(e2 - s2)
