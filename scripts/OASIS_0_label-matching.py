import os
import re
import pandas as pd
import numpy as np


def path2key(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    id, day = re.findall(r'OAS\d+|d\d+', filename)
    key = f"{id}_{day}"
    return key


def cdr2dx(cdr):
    if cdr == 0.0:
        return 'NL'
    elif cdr == 0.5:
        return 'MCI'
    elif cdr >= 1.0:
        return 'AD'
    else:
        raise ValueError

# Data directory
DATA_DIR = 'E:/data/OASIS'

# Read files
match_info_mri = pd.read_excel(os.path.join(DATA_DIR, 'OASIS_PIB-AV45-MR-matched.xlsx'), sheet_name='MR only')
match_info_pib = pd.read_excel(os.path.join(DATA_DIR, 'OASIS_PIB-AV45-MR-matched.xlsx'), sheet_name='PIB-MR')
match_info_av45 = pd.read_excel(os.path.join(DATA_DIR, 'OASIS_PIB-AV45-MR-matched.xlsx'), sheet_name='AV45-MR')

clinical_info = pd.read_csv(os.path.join(DATA_DIR, 'OASIS3 Clinical Data_0815.csv'))

# Step 1. assign labels using clinical_info
# 1-1. clinical day
clinical_info['Clinical Day'] = clinical_info['ADRC_ADRCCLINICALDATA ID'].str.extract('d(\d{4})').astype(int)

# 1-2. convert cdr into DX (diagnosis)
clinical_info['DX'] = clinical_info['cdr'].apply(cdr2dx)

# 1-3. assign converter (C) and non-converter (NC) labels for 18 months / 36 months time window
clinical_info = clinical_info.sort_values(['Subject', 'Clinical Day']).reset_index(drop=True)
subjects = clinical_info['Subject'].unique()

# for simplicity, one month is equal to 30 days.
clinical_info['Label'] = None

def label_rows(subject_rows):
    for index, row in subject_rows.iterrows():
        if row['DX'] == 'MCI':
            # 36 months = 1080 days
            future_rows_within_36_months = subject_rows[(subject_rows['Clinical Day'] > row['Clinical Day']) &
                                                        (subject_rows['Clinical Day'] <= row['Clinical Day'] + 1080)]
            future_rows_after_36_months = subject_rows[subject_rows['Clinical Day'] > row['Clinical Day'] + 1080]

            if 'AD' in future_rows_within_36_months['DX'].values:
                clinical_info.at[index, 'Label'] = 'C'
            elif not future_rows_after_36_months.empty:
                clinical_info.at[index, 'Label'] = 'NC'
            # if there is no record after 36 months, label is still None (cannot be determined)

# iterate over all subjects
subjects = clinical_info['Subject'].unique()
for subject in subjects:
    subject_rows = clinical_info[clinical_info['Subject'] == subject]
    label_rows(subject_rows)

# rename for unification
clinical_info = clinical_info.rename(columns={'Subject': 'OASIS ID'})

# Step 2. Concatenate matching spreadsheet
def transform_path(path, tracer):
    if tracer == 'mri':
        # Extract the desired parts from the path using regex
        match = re.search(r'(OAS\d+).*?(d\d+).*?(anat\d+)', path)
        if match:
            return "_".join(match.groups())
    return path

# 2-1. Convert Path into a format match with folder name
# MRI Path
match_info_mri['MR Path'] = match_info_mri['MR Path'].apply(lambda x: transform_path(x, 'mri'))
match_info_pib['MR Path'] = match_info_pib['MR Path'].apply(lambda x: transform_path(x, 'mri'))
match_info_av45['MR Path'] = match_info_av45['MR Path'].apply(lambda x: transform_path(x, 'mri'))

# PET Path
match_info_pib['PIB Path'] = match_info_pib['session_id']
match_info_av45['AV45 Path'] = match_info_av45['session_id']

# Concatenate
match_info_av45 = match_info_av45.rename(columns={'OASIS_ ID': 'OASIS ID', 'AV45 day': 'AV45 Day'})
match_info_pib = match_info_pib.rename(columns={'PIB day': 'PIB Day'})

match_info_mri['Tracer'] = 'MRI'

match_info = pd.concat([match_info_mri[['OASIS ID', 'MR Day', 'MR Path', 'Tracer']],
                        match_info_pib[['OASIS ID', 'MR Day', 'PIB Day', 'MR Path', 'PIB Path', 'Tracer']],
                        match_info_av45[['OASIS ID', 'MR Day', 'AV45 Day', 'MR Path', 'AV45 Path', 'Tracer']]],
                       ignore_index=True)

match_info = match_info.sort_values(['OASIS ID', 'MR Day']).reset_index(drop=True)

# Step 3. Aggregate all information. Use demographic spreadsheet as the base
data_info = pd.read_excel(os.path.join(DATA_DIR, 'OA006-MR-OASIS-demo-ingo.xlsx'),
                          sheet_name='OA006_MR_OASIS_demo_ingo')
data_info = data_info.rename(columns={'Clinical day': 'Clinical Day'})

data_info = data_info.merge(match_info, on=['OASIS ID', 'MR Day'], how='left')
data_info = data_info.merge(clinical_info[['OASIS ID', 'Clinical Day', 'mmse', 'Label']],
                            on=['OASIS ID', 'Clinical Day'], how='left')

# Step 4. For simplicity, remove redundant columns, replace some column names, and others
# Try to match with ADNI
data_info = data_info.drop(['OASIS_Session_ID'], axis=1)
data_info = data_info.rename(columns={'AgeatEntry': 'Age',
                                      'GENDER ( 1=male, 2=female)': 'PTGENDER',
                                      'cdr': 'CDGLOBAL',
                                      'mmse': 'MMSCORE'})
data_info['PTGENDER'] = data_info['PTGENDER'] - 1


# Step 5. Create APOE Status
# Rule 1: use apoe values. if zero 4, NL. if one 4, HT. if two 4, HM.
# Rule 2: convert NL to 0, otherwise 1
# Final Rule: if there is any 4 in apoe, APOE Status becomes 1. Otherwise, 0.
data_info['APOE Status'] = data_info['apoe'].apply(lambda x: 1 if '4' in str(x) else 0)

data_info['MR Path'] = data_info['MR Path'].fillna('')
data_info['PIB Path'] = data_info['PIB Path'].fillna('')
data_info['AV45 Path'] = data_info['AV45 Path'].fillna('')


# Step 6. Save file
data_info.to_csv(os.path.join(DATA_DIR, 'data_info.csv'), index=False)

# Step 7. IS_FILE (image)
data_info_ = data_info.copy()
data_info_ = data_info_.loc[data_info_['Label'].isin(['C', 'NC'])]

mri_paths = data_info_['MR Path'].tolist()
pib_paths = data_info_['PIB Path'].tolist()
av45_paths = data_info_['AV45 Path'].tolist()

mri_image_paths = os.listdir(os.path.join(DATA_DIR, 'MRI'))
pib_image_paths = os.listdir(os.path.join(DATA_DIR, 'PIB'))
av45_image_paths = os.listdir(os.path.join(DATA_DIR, 'AV45'))

set(mri_paths) - set(mri_image_paths)
set(pib_paths) - set(pib_image_paths)
set(av45_paths) - set(av45_image_paths)
