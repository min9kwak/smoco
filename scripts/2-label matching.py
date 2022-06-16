import os
import glob
import pandas as pd
import numpy as np
import re
import datetime

DATA_DIR = "D:/data/ADNI"

#
columns = ['RID', 'DX', 'Conv', 'Filename']
demo_columns = ['PTGENDER (1=male, 2=female)', 'Age', 'PTEDUCAT', 'APOE Status', 'MMSCORE', 'CDGLOBAL', 'SUM BOXES']
columns = columns + demo_columns

mri_labels = pd.read_csv(os.path.join(DATA_DIR, "labels/MRI_features_all.csv"))
mri_labels = mri_labels[columns]

#
summary = mri_labels.copy()
summary['Filename'].str.split('/')
summary[['PET Type', 'MRI']] = summary['Filename'].str.split('/', expand=True)
summary = summary.loc[~summary.MRI.isna()].reset_index(drop=True)

# add PET
pet_proc_params = glob.glob(os.path.join(DATA_DIR, "PUP_FBP/*/*/*.param"), recursive=True)
logs = []
for pet_proc_param in pet_proc_params:
    with open(pet_proc_param, 'rb') as f:
        res = f.readlines()
        r = [r for r in res if r.startswith(b'fsdir=/datadrive/ADNI_STTR/proc/FS7')]
        if len(r) == 1:
            k = pet_proc_param.split('\\')[-1].replace('_pet.param', '')
            v = r[0].decode('utf-8').split('/')[-2]
            logs.append([k, v])
        else:
            raise ValueError
logs = pd.DataFrame(logs)
logs.columns = ['PET', 'MRI']

summary = pd.merge(left=summary, right=logs, how='left', left_on=['MRI'], right_on=['MRI'])
summary.Conv = summary.Conv.fillna(-1)
summary.DX = summary.DX.fillna(-1)
summary['MCI'] = [1 if d in [2, 4, 8] else 0 for d in summary.DX.tolist()]

summary.Conv = summary.Conv.astype(int)
summary.to_csv(os.path.join(DATA_DIR, "labels/data_info.csv"), index=False)
