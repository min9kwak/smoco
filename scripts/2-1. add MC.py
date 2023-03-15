import pandas as pd


data_info = pd.read_csv('D:/data/ADNI/labels/data_info.csv', converters={'RID': str, 'Conv': int})
data_info = data_info.loc[data_info.IS_FILE]

mc_table = pd.read_excel('D:/data/ADNI/labels/AV45_FBP_SUVR.xlsx', sheet_name='list_id_SUVR_RSF')

data_pet_id = data_info['PET'].loc[~data_info.PET.isna()]
temp = pd.merge(left=data_pet_id, right=mc_table[['ID', 'MC']], left_on=['PET'], right_on=['ID'])
(temp['ID'].isna()).sum()
 
for i, row in temp.iterrows():
    assert row['PET'] == row['ID']

