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


# TODO: concatenate AIBL_01032023.xlsx and AIBL_labels.csv with exist_file column
root = 'D:/data/AIBL'
data_info = 'AIBL_01032023.xlsx'
data_labels = 'AIBL_labels.csv'
image_dir = 'template/PIB'

data_info = pd.read_excel(os.path.join(root, data_info), sheet_name='PIB-MR', converters={'RID': str})
data_info['Month'] = data_info['Visit'].apply(convert_month)
data_labels = pd.read_csv(os.path.join(root, data_labels), converters={'RID': str})

data_info[['Conv_18', 'Conv_36']] = np.nan
data_info['is_file'] = False
data_info['image_file'] = ''

for i, row in tqdm.tqdm(data_info.iterrows(), total=len(data_info)):

    rid, visit, month = row['RID'], row['Visit'], row['Month']
    index = (data_labels['RID'] == rid) & (data_labels['Month'] == month)
    assert sum(index) == 1

    result = data_labels.loc[index]
    data_info.loc[i, 'Conv_18'] = result['Conv_18'].item()
    data_info.loc[i, 'Conv_36'] = result['Conv_36'].item()

    pib_file = os.path.join(root, image_dir, f'{rid}{visit}.pkl')
    is_file = os.path.isfile(pib_file)
    data_info.loc[i, 'is_file'] = is_file
    data_info.loc[i, 'image_file'] = f'{rid}{visit}.pkl'

data_info.to_csv(os.path.join(root, 'data_info.csv'), index=False)
