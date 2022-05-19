import os
import pandas as pd
import nibabel as nib

root = 'D:/data/ADNI'
data_info = 'labels/data_info.csv'
data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'MONTH': int, 'Conv': int})
data_info = data_info[data_info['Conv'].isin([0, 1])]

data_files = [(os.path.join(root, 'FS7', row.MRI, 'mri/brain.mgz'))
               for _, row in data_info.iterrows()]

import pickle
import tqdm

for image_file in tqdm.tqdm(data_files):
    image = nib.load(image_file)
    image = nib.as_closest_canonical(image)
    image = image.get_fdata()
    with open(image_file.replace('.mgz', '.pkl'), 'wb') as f:
        pickle.dump(image, f)
