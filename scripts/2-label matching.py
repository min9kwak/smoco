import os
import glob
import pandas as pd
import numpy as np
import re
import datetime

DATA_DIR = "D:/data/ADNI"

def viscode2month(viscode):
    if type(viscode) != str:
        if np.isnan(viscode):
            return -1
    elif viscode.startswith('m'):
        return int(viscode[1:])
    elif viscode == 'scmri':
        return 0
    else:
        raise ValueError

# MRI demographic
mri_filenames = [os.path.join(DATA_DIR, "source files/FS7/gzlista.txt"),
                 os.path.join(DATA_DIR, "source files/FS7/gzlistb.txt")]
mri_filenames = pd.concat([pd.read_table(f, header=None) for f in mri_filenames]).values.reshape(-1, ).tolist()

mri_demo = os.path.join(DATA_DIR, "labels/312-FS outputs-wSubjectinfo MRI.xlsx")
mri_demo = pd.read_excel(mri_demo, sheet_name=0, header=0, converters={'RID': str})
mri_demo = mri_demo.rename(columns={'rh aparc thickness': 'Filename'})

mri_demo['VISCODE2'] = mri_demo['VISCODE2'].fillna('scmri')

mri_demo[["PET_TYPE", "MRI"]] = mri_demo["Filename"].str.split("/", expand=True)

mri_labels = os.path.join(DATA_DIR, "labels/DX_USE.csv")
mri_labels = pd.read_csv(mri_labels, converters={'RID': str})
mri_labels['VISCODE2'] = mri_labels['VISCODE2'].replace({'bl': 'scmri', 'sc': 'scmri'})

##
summary = pd.DataFrame()
summary['MRI Filenames'] = mri_filenames
summary['MRI'] = summary['MRI Filenames'].apply(lambda x: x.replace(".tar.gz", ""))
print('image file list only - ', len(summary))

summary = pd.merge(left=summary, right=mri_demo[['MRI', 'Date', 'RID', 'VISCODE2']], how='left',
                   left_on='MRI', right_on='MRI')
print('image file & mri demo - ', len(summary))

print('not found - ', summary.RID.isna().sum())
summary = summary.dropna(subset=['RID'])
print('not found after dropped missing RID - ', summary.RID.isna().sum())

# convert VISCODE2 to MONTH
summary['MONTH'] = summary.VISCODE2.apply(viscode2month)
mri_demo['MONTH'] = mri_demo.VISCODE2.apply(viscode2month)
mri_labels['MONTH'] = mri_labels.VISCODE2.apply(viscode2month)

# co-register PET
pet_filenames = os.path.join(DATA_DIR, "source files/PUP_FBP/gzlist.txt")
pet_filenames = pd.read_table(pet_filenames, header=None).values.reshape(-1, ).tolist()

pet_demo = os.path.join(DATA_DIR, "labels/AV45_FBP_SUVR.xlsx")
pet_demo = pd.read_excel(pet_demo, sheet_name=0, header=0, converters={'RID': str})

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

# 2. check missing information
# brain.mgz
brain_files = glob.glob(os.path.join(DATA_DIR, "FS7/*/mri/brain.mgz"), recursive=True)
brain_files = [b.split('\\')[-3] for b in brain_files]
summary['brain'] = summary['MRI'].apply(lambda x: 1 if x in brain_files else 0)

# wmparc.mgz
wmparc_files = glob.glob(os.path.join(DATA_DIR, "FS7/*/mri/wmparc.mgz"), recursive=True)
wmparc_files = [w.split('\\')[-3] for w in wmparc_files]
summary['wmparc'] = summary['MRI'].apply(lambda x: 1 if x in wmparc_files else 0)

# 3. Assign labels (converters vs. non-converters)
# drop columns with out RID and VISCODE2
def get_converter_status(features, time_window=36):
    rid_all = features['RID'].unique()
    features = features.sort_values(by=['RID', 'MONTH'])
    rid_mci = []
    month_mci = []
    det_t = []
    conv_end = []
    count_mci = 0
    count_mci_1 = 0
    count_mci_2 = 0

    for i, rid in enumerate(rid_all):

        patient_rec = features[features['RID'] == rid]
        if np.any(np.isin(np.array(patient_rec['DX']), [2, 4, 8])):

            end_flag = 0
            conv_flag = np.zeros((20,), dtype=np.int32)
            month_start = np.zeros((20,))
            mci_num = 0
            start_flag = 0

            diag = np.array(patient_rec['DX'])
            mon = np.array(patient_rec['MONTH'])
            for j in range(len(diag)):

                if start_flag == 0:
                    mci_def = [2, 4, 8]
                else:
                    mci_def = [2, 4, 8]

                if np.isin(diag[j], mci_def):

                    conv_flag[mci_num] = 1
                    month_start[mci_num] = mon[j]
                    mci_num += 1
                    start_flag = 1

                elif np.isin(diag[j], [3, 5, 6]) and np.sum(conv_flag) > 0:
                    end_flag = 1
                    for k in range(np.sum(conv_flag)):
                        rid_mci.append(rid)
                        month_mci.append(month_start[k])
                        det_t.append(mon[j] - month_start[k])
                        if det_t[-1] <= time_window:

                            conv_end.append(1)
                            count_mci_1 += 1
                        else:
                            conv_end.append(0)
                            count_mci_2 += 1

                    break

            if end_flag == 0:
                for k in range(np.sum(conv_flag)):
                    if mon[j] > month_start[k]:
                        rid_mci.append(rid)
                        month_mci.append(month_start[k])
                        det_t.append(mon[j] - month_start[k])
                        conv_end.append(0)
                        if det_t[-1] >= time_window:
                            count_mci_2 += 1
                        else:
                            rid_mci.pop()
                            month_mci.pop()
                            det_t.pop()
                            conv_end.pop()
                            break
            else:
                count_mci += 1

    mci_summ = pd.DataFrame(np.c_[rid_mci, month_mci, det_t, conv_end])
    mci_summ.columns = ['RID', 'MONTH', 'Delta_t', 'Conv']

    print("conv pat:", count_mci, "conv rec:", count_mci_1, "non-conv rec:", count_mci_2)
    print("Total pat:", len(np.unique(rid_mci)))

    return mci_summ

mci = get_converter_status(mri_labels)

mci['RID'] = mci['RID'].apply(str)
mci['MONTH'] = mci['MONTH'].apply(float).apply(int)

summary['RID'] = summary['RID'].apply(str)
summary['MONTH'] = summary['MONTH'].apply(int)
summary = pd.merge(summary, mci, how='left', on=['RID', 'MONTH'])
summary = summary.drop_duplicates(subset=['RID', 'MONTH'])

summary['Conv'] = summary['Conv'].fillna(-1).apply(int)

# DX & MCI-related
summary = pd.merge(summary, mri_labels[['RID', 'MONTH', 'DX']], how='left', on=['RID', 'MONTH'])
summary['MCI'] = summary['DX'].apply(lambda x: 1 if x in [2, 4, 6] else 0)
for i, row in summary.iterrows():
    if row.Conv != -1:
        summary.loc[i, 'MCI'] = 1

# TODO: add clinical/demographic features
demo_feature_names = ['']
summary = pd.merge(summary, mri_demo[['RID', 'MONTH'] + demo_feature_names],
                   how='left', on=['RID', 'MONTH'])

summary.to_csv(os.path.join(DATA_DIR, "labels/data_info.csv"), index=False)

print('# non-converters: ', sum(summary.Conv == 0))
print('#     converters: ', sum(summary.Conv == 1))
print('#          total: ', sum(summary.Conv != -1))
