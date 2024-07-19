import os
import re
import pandas as pd

def transform_path(path, tracer):
    if tracer == 'mri':
        # Extract the desired parts from the path using regex
        match = re.search(r'(OAS\d+).*?(d\d+).*?(anat\d+)', path)
        if match:
            return "_".join(match.groups())
    return path


root = 'D:/data/'
data_info_aibl = os.path.join(root, 'AIBL/data_info.csv')
data_info_aibl = pd.read_csv(data_info_aibl)

# preprocessing oasis
data_label_oasis = pd.read_csv("E:/data/OASIS/OASIS_all_labels.csv")
data_label_oasis = data_label_oasis.rename(columns={'cdr': 'CDGLOBAL',
                                                    'mmse': 'MMSCORE',
                                                    'GENDER ( 1=male, 2=female)': 'PTGENDER',
                                                    'APOE4 Status': 'APOE',
                                                    'Clinical day': 'Clinical Day',
                                                    'OASIS_ID': 'OASIS ID'})
data_label_oasis = data_label_oasis[['OASIS ID', 'MR Day', 'Clinical Day', 'Tracer',
                                     'Age', 'APOE', 'MMSCORE', 'CDGLOBAL', 'PTGENDER',
                                     'label_m1', 'label_m2', 'label_m3']]

match_info_mri = pd.read_excel(os.path.join('E:/data/OASIS', 'OASIS_PIB-AV45-MR-matched.xlsx'), sheet_name='MR only')
match_info_pib = pd.read_excel(os.path.join('E:/data/OASIS', 'OASIS_PIB-AV45-MR-matched.xlsx'), sheet_name='PIB-MR')
match_info_av45 = pd.read_excel(os.path.join('E:/data/OASIS', 'OASIS_PIB-AV45-MR-matched.xlsx'), sheet_name='AV45-MR')

match_info_mri['MR Path'] = match_info_mri['MR Path'].apply(lambda x: transform_path(x, 'mri'))
match_info_pib['MR Path'] = match_info_pib['MR Path'].apply(lambda x: transform_path(x, 'mri'))
match_info_av45['MR Path'] = match_info_av45['MR Path'].apply(lambda x: transform_path(x, 'mri'))
match_info_pib['PIB Path'] = match_info_pib['session_id']
match_info_av45['AV45 Path'] = match_info_av45['session_id']
match_info_av45 = match_info_av45.rename(columns={'OASIS_ ID': 'OASIS ID', 'AV45 day': 'AV45 Day'})
match_info_pib = match_info_pib.rename(columns={'PIB day': 'PIB Day'})
match_info_mri['Tracer'] = 'MRI'

match_info = pd.concat([match_info_mri[['OASIS ID', 'MR Day', 'MR Path', 'Tracer']],
                        match_info_pib[['OASIS ID', 'MR Day', 'PIB Day', 'MR Path', 'PIB Path', 'Tracer']],
                        match_info_av45[['OASIS ID', 'MR Day', 'AV45 Day', 'MR Path', 'AV45 Path', 'Tracer']]],
                       ignore_index=True)
match_info = match_info.sort_values(['OASIS ID', 'MR Day']).reset_index(drop=True)

data_label_oasis = data_label_oasis.merge(match_info, on=['OASIS ID', 'MR Day'], how='left')

data_label_oasis['PTGENDER'] = data_label_oasis['PTGENDER'] - 1

data_label_oasis['MR Path'] = data_label_oasis['MR Path'].fillna('')
data_label_oasis['PIB Path'] = data_label_oasis['PIB Path'].fillna('')
data_label_oasis['AV45 Path'] = data_label_oasis['AV45 Path'].fillna('')

data_label_oasis = data_label_oasis.replace({'Label': {0: 'NC', 1: 'C'}})

data_label_oasis.to_csv('E:/data/OASIS/data_info_all.csv', index=False)






data_info_oasis = os.path.join(root, 'OASIS/data_info.csv')

data_info_oasis = pd.read_csv(data_info_oasis)

# Make AIBL and OASIS similar to ADNI
# change AIBL
data_info_aibl.loc[data_info_aibl['IS_MRI_FILE'], 'MR Path'] = data_info_aibl.loc[data_info_aibl['IS_MRI_FILE'], 'image_file']
data_info_aibl.loc[data_info_aibl['IS_PIB_FILE'], 'PIB Path'] = data_info_aibl.loc[data_info_aibl['IS_PIB_FILE'], 'image_file']

# TODO:
data_info_aibl['AV45 Path'] = ''
data_info_aibl['IS_AV45_FILE'] = False


data_info_aibl.loc[:, 'APOE'] = data_info_aibl['APOE'].fillna('NC')
data_info_aibl['APOE'] = [0 if a == 'NC' else 1 for a in data_info_aibl['APOE'].values]
data_info_aibl['PTGENDER'] = data_info_aibl['PTGENDER'] - 1

data_info_aibl = data_info_aibl.rename(columns={'Conv_36': 'Conv'})

# change OASIS
data_info_oasis = data_info_oasis.rename(columns={'APOE Status': 'APOE'})
data_info_oasis = data_info_oasis.rename(columns={'OASIS ID': 'RID', 'Label': 'Conv'})

data_info_oasis['MR Path'] = data_info_oasis['MR Path'] + '.pkl'
data_info_oasis['PIB Path'] = data_info_oasis['PIB Path'] + '.pkl'
data_info_oasis['AV45 Path'] = data_info_oasis['AV45 Path'] + '.pkl'


# Concatenate
info_columns = ['RID', 'Age', 'PTGENDER', 'CDGLOBAL', 'MMSCORE', 'APOE', 'Conv', 'MR Path', 'PIB Path', 'AV45 Path', 'Source']
data_info_aibl['Source'] = 'AIBL'
data_info_oasis['Source'] = 'OASIS'

data_info = pd.concat([data_info_aibl[info_columns], data_info_oasis[info_columns]], axis=0).reset_index(drop=True)

data_info.to_csv('D:/data/aibl_oasis_data_info.csv', index=False)
