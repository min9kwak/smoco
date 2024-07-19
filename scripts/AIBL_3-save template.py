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


root = 'E:/data/AIBL'
data_info = 'AIBL_01032023.xlsx'
data_labels = 'AIBL_labels.csv'

data_info = pd.read_excel(os.path.join(root, data_info), sheet_name='PIB-MR', converters={'RID': str})
data_info['Month'] = data_info['Visit'].apply(convert_month)
data_labels = pd.read_csv(os.path.join(root, data_labels), converters={'RID': str})

missing_mri = []
missing_pib = []

pib_save_dir = 'D:/data/AIBL/template/PIB'
os.makedirs(pib_save_dir, exist_ok=True)

for i, row in tqdm.tqdm(data_info.iterrows(), total=len(data_info)):

    rid, visit = row['RID'], row['Visit']
    mri_path, pib_path = row['MR Path'], row['PIB Path']

    # MRI - brainmask
    loc = mri_path.split('/')[6]
    if visit == 'bl':
        mri_path = os.path.join('MRI', f'{rid}{visit}')
    else:
        mri_path = os.path.join('MRI', f'{rid}_{visit}_{loc}')
    mri_path = os.path.join(root, mri_path)

    # PIB - scan
    pib_path = os.path.join('PIB', f'{rid}{visit}')
    pib_path = os.path.join(root, pib_path)

    if not os.path.exists(mri_path):
        continue
    if not os.path.exists(pib_path):
        continue

    mri_path = os.path.join(mri_path, 'cat12/mri')
    mri_file = os.listdir(mri_path)[0]
    assert mri_file.startswith('wm') and mri_file.endswith('.nii')
    mri_file = os.path.join(mri_path, mri_file)

    pib_path = os.path.join(pib_path, 'pet_proc')
    pib_file = [f for f in os.listdir(pib_path) if f.startswith('w_') and f.endswith('.nii')][0]
    pib_file = os.path.join(pib_path, pib_file)

    # open
    mri = nib.load(mri_file).get_fdata()
    mri_pad = np.pad(mri, ((12, 12), (0, 0), (12, 12)), 'constant')

    pib = nib.load(pib_file).get_fdata()
    pib = np.nan_to_num(pib)
    mask = (mri == 0)
    pib[mask] = 0
    pib_pad = np.pad(pib, ((12, 12), (0, 0), (12, 12)), 'constant')

    pib_id = pib_path.split('\\')[2]
    with open(os.path.join(pib_save_dir, f'{pib_id}.pkl'), 'wb') as fb:
        pickle.dump(pib_pad, fb)
