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
mri_save_dir = 'D:/data/AIBL/template/MRI'
os.makedirs(mri_save_dir, exist_ok=True)

# concatenate all spreadsheets
# 1. PIB-MR
base_columns = ['PTGENDER', 'Status', 'DX', 'MMSCORE', 'CDGLOBAL']

# PIB-MR
df_pib = pd.read_excel(os.path.join(root, data_info), sheet_name="PIB-MR")
df_pib = df_pib[['RID', 'Visit', 'PIB Path', 'MR Path', 'AgeatPIBScan'] + base_columns]
df_pib = df_pib.rename(columns={'PIB Path': 'pib_path', 'MR Path': 'mri_path', 'AgeatPIBScan': 'Age'})
df_pib = df_pib.astype({'RID': str})
df_pib['Month'] = df_pib['Visit'].apply(convert_month)

# AV45-MR
df_av45 = pd.read_excel(os.path.join(root, data_info), sheet_name="AV45-MR")
df_av45 = df_av45[['RID', 'visit_code', 'av45_path', 'Path', 'AgeatAV45Scan'] + base_columns]
df_av45 = df_av45.rename(columns={'visit_code': 'Visit', 'Path': 'mri_path', 'AgeatAV45Scan': 'Age'})
df_av45 = df_av45.astype({'RID': str})
df_av45['Month'] = df_av45['Visit'].apply(convert_month)

# Flute-MR
df_flute = pd.read_excel(os.path.join(root, data_info), sheet_name="Flute-MR")
df_flute = df_flute[['RID', 'visit_code', 'flute_path', 'Path', 'AgeatFluteScan'] + base_columns]
df_flute = df_flute.rename(columns={'visit_code': 'Visit', 'Path': 'mri_path', 'AgeatFluteScan': 'Age'})
df_flute = df_flute.astype({'RID': str})
df_flute['Month'] = df_flute['Visit'].apply(convert_month)

# MR Only
df_mri = pd.read_excel(os.path.join(root, data_info), sheet_name="MR only")
df_mri = df_mri[['Subject ID', 'VC', 'MR path', 'AgeAtMRI'] + base_columns]
df_mri = df_mri.rename(columns={'Subject ID': 'RID', 'VC': 'Visit', 'MR path': 'mri_path', 'AgeAtMRI': 'Age'})
df_mri = df_mri.astype({'RID': str})
df_mri['Month'] = df_mri['Visit'].apply(convert_month)

# data_labels
data_labels = pd.read_csv(os.path.join(root, data_labels), converters={'RID': str})

missing_mri = []

flag = False
for i, row in tqdm.tqdm(data_labels.iterrows(), total=len(data_labels)):
    rid, month, visit = row['RID'], row['Month'], row['Visit']
    for df_type, df in zip(['pib', 'av45', 'flute', 'mri'], [df_pib, df_av45, df_flute, df_mri]):
        record = df.loc[(df['RID'] == rid) & (df['Month'] == month)]

        if not record.empty:
            assert not flag
            mri_path_ = record['mri_path'].item()
            loc = mri_path_.split('/')[6]
            if visit == 'bl':
                mri_path = os.path.join('MRI', f'{rid}{visit}')
            else:
                mri_path = os.path.join('MRI', f'{rid}_{visit}_{loc}')
            mri_path = os.path.join(root, mri_path)
            flag = True

            # scan
            if not os.path.exists(mri_path):
                missing_mri.append(mri_path)
                continue
            mri_path = os.path.join(mri_path, 'cat12/mri')
            mri_file = os.listdir(mri_path)[0]
            assert mri_file.startswith('wm') and mri_file.endswith('.nii')
            mri_file = os.path.join(mri_path, mri_file)

            # open
            mri = nib.load(mri_file).get_fdata()
            mri_pad = np.pad(mri, ((12, 12), (0, 0), (12, 12)), 'constant')
            assert not np.isnan(mri_pad).any()

            mri_id = mri_path.split('\\')[2]
            assert not os.path.exists(os.path.join(mri_save_dir, f'{rid}{visit}.pkl'))
            with open(os.path.join(mri_save_dir, f'{rid}{visit}.pkl'), 'wb') as fb:
                pickle.dump(mri_pad, fb)

        else:
            flag = False
    flag = False
