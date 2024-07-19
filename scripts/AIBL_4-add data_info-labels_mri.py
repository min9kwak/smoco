import os
import pickle
import numpy as np
import pandas as pd
import tqdm
import nibabel as nib


def convert_month(month_str):
    if month_str == 'bl':
        return 0
    if month_str.startswith('m'):
        return int(month_str[1:])
    return None


# TODO: add clinical and demographic information
root = 'D:/data/AIBL'
data_labels = 'AIBL_labels.csv'
image_dir = 'template/MRI'

data_labels = pd.read_csv(os.path.join(root, data_labels), converters={'RID': str})
data_labels['is_file'] = False
data_labels['image_file'] = ''

for i, row in tqdm.tqdm(data_labels.iterrows(), total=len(data_labels)):

    rid, visit, month = row['RID'], row['Visit'], row['Month']
    index = (data_labels['RID'] == rid) & (data_labels['Month'] == month)
    assert sum(index) == 1

    mri_file = os.path.join(root, image_dir, f'{rid}{visit}.pkl')
    is_file = os.path.isfile(mri_file)
    data_labels.loc[i, 'is_file'] = is_file
    data_labels.loc[i, 'image_file'] = f'{rid}{visit}.pkl'

data_labels.to_csv(os.path.join(root, 'data_info_mri.csv'), index=False)
