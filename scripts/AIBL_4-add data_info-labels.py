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


# Load data information and labels
root = 'D:/data/AIBL'
mri_dir = 'template/MRI'
pib_dir = 'template/PIB'

data_info = 'AIBL_01032023.xlsx'
data_labels = 'AIBL_labels.csv'
data_labels = pd.read_csv(os.path.join(root, data_labels), converters={'RID': str})

data_labels['IS_MRI_FILE'] = False
data_labels['IS_PIB_FILE'] = False

data_labels['image_file'] = data_labels['RID'] + data_labels['Visit'] + '.pkl'
for i, row in data_labels.iterrows():
    if os.path.isfile(os.path.join(root, mri_dir, row['image_file'])):
        data_labels.loc[i, 'IS_MRI_FILE'] = True
    if os.path.isfile(os.path.join(root, pib_dir, row['image_file'])):
        data_labels.loc[i, 'IS_PIB_FILE'] = True

data_labels.to_csv('D:/data/AIBL/data_info.csv', index=False)
