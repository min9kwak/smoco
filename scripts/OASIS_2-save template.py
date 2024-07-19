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


root = 'E:/data/OASIS'
# data_info = 'data_info.csv'
data_info = pd.read_csv(os.path.join("D:/", 'data/', 'aibl_oasis_data_info_all.csv'))
# data_info = data_info.loc[data_info['Label'].isin(['C', 'NC'])]
data_info = data_info.loc[data_info['Source'] == 'OASIS']

# failed PIB
data_info['PIB Path'] = data_info['PIB Path'].replace('OAS30896_PIB_d1601.pkl', np.nan)

mri_save_dir = 'D:/data/OASIS/template/MRI'
pib_save_dir = 'D:/data/OASIS/template/PIB'
av45_save_dir = 'D:/data/OASIS/template/AV45'
os.makedirs(mri_save_dir, exist_ok=True)
os.makedirs(pib_save_dir, exist_ok=True)
os.makedirs(av45_save_dir, exist_ok=True)


for i, row in tqdm.tqdm(data_info.iterrows(), total=len(data_info)):

    mri_name, pib_name, av45_name = row['MR Path'], row['PIB Path'], row['AV45 Path']

    mri_name = mri_name.replace('.pkl', '')

    # 1. MRI
    mri_path = os.path.join(root, 'MRI', mri_name, 'cat12/mri')
    mri_file = os.listdir(mri_path)[0]
    assert mri_file.startswith('wm') and mri_file.endswith('.nii')
    mri_file = os.path.join(mri_path, mri_file)

    # Load MRI
    mri = nib.load(mri_file).get_fdata()
    mri_pad = np.pad(mri, ((12, 12), (0, 0), (12, 12)), 'constant')
    assert not np.isnan(mri_pad).any()

    # Save
    with open(os.path.join(mri_save_dir, f'{mri_name}.pkl'), 'wb') as fb:
        pickle.dump(mri_pad, fb)

    # 2. PIB
    if not type(pib_name) == float:

        pib_name = pib_name.replace('.pkl', '')

        pib_path = os.path.join(root, 'PIB', pib_name, 'pet_proc')
        pib_file = [f for f in os.listdir(pib_path) if f.startswith('w_') and f.endswith('.nii')][0]
        pib_file = os.path.join(pib_path, pib_file)

        pib = nib.load(pib_file).get_fdata()
        pib = np.nan_to_num(pib)
        mask = (mri == 0)
        pib[mask] = 0
        pib_pad = np.pad(pib, ((12, 12), (0, 0), (12, 12)), 'constant')

        with open(os.path.join(pib_save_dir, f'{pib_name}.pkl'), 'wb') as fb:
            pickle.dump(pib_pad, fb)

    # 3. AV45
    if not type(av45_name) == float:

        av45_name = av45_name.replace('.pkl', '')

        av45_path = os.path.join(root, 'AV45', av45_name, 'pet_proc')
        av45_file = [f for f in os.listdir(av45_path) if f.startswith('w_') and f.endswith('.nii')][0]
        av45_file = os.path.join(av45_path, av45_file)

        av45 = nib.load(av45_file).get_fdata()
        av45 = np.nan_to_num(av45)
        mask = (mri == 0)
        av45[mask] = 0
        av45_pad = np.pad(av45, ((12, 12), (0, 0), (12, 12)), 'constant')

        with open(os.path.join(av45_save_dir, f'{av45_name}.pkl'), 'wb') as fb:
            pickle.dump(av45_pad, fb)
