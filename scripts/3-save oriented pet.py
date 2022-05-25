import os
import pandas as pd
import nibabel as nib

root = 'D:/data/ADNI'
data_info = 'labels/data_info.csv'
data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'MONTH': int, 'Conv': int})
data_info = data_info[~data_info.PET.isna()]

data_files = [(os.path.join(root, 'FS7', row.MRI, 'mri/brainmask.mgz'),
               os.path.join(root, 'PUP_FBP', row.PET, f'pet_proc/{row.PET}_SUVR.nii.gz'))
              for _, row in data_info.iterrows()]

to_dir = os.path.join(root, 'PUP_FBP_oriented')
os.makedirs(to_dir, exist_ok=True)

import pickle
import tqdm

for mask_file, image_file in tqdm.tqdm(data_files):
    mask, image = nib.load(mask_file), nib.load(image_file)
    mask, image = nib.as_closest_canonical(mask), nib.as_closest_canonical(image)
    mask, image = mask.get_fdata(), image.get_fdata().squeeze()
    mask = (mask == 0)
    image[mask] = 0

    with open(image_file.replace('nii.gz', 'pkl'), 'wb') as f:
        pickle.dump(image, f)
