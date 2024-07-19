import os
import pandas as pd


root = 'D:/data/'
data_info_aibl = os.path.join(root, 'AIBL/data_info.csv')
data_info_oasis = os.path.join(root, 'OASIS/data_info.csv')

data_info_aibl = pd.read_csv(data_info_aibl)
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
