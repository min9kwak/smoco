import pandas as pd
import numpy as np


def convert_month(month_str):
    if month_str == 'bl':
        return 0
    if month_str.startswith('m'):
        return int(month_str[1:])
    return None

# read file
filename = "E:/data/AIBL/AIBL_01032023.xlsx"
base_columns = ['PTGENDER', 'Status', 'DX', 'MMSCORE', 'CDGLOBAL']

# PIB-MR
df_pib = pd.read_excel(filename, sheet_name="PIB-MR")
df_pib = df_pib[['RID', 'Visit', 'PIB Path', 'MR Path', 'AgeatPIBScan'] + base_columns]
df_pib = df_pib.rename(columns={'PIB Path': 'pib_path', 'MR Path': 'mri_path', 'AgeatPIBScan': 'Age'})
df_pib = df_pib.astype({'RID': str})
df_pib['Month'] = df_pib['Visit'].apply(convert_month)

# AV45-MR
df_av45 = pd.read_excel(filename, sheet_name="AV45-MR")
df_av45 = df_av45[['RID', 'visit_code', 'av45_path', 'Path', 'AgeatAV45Scan'] + base_columns]
df_av45 = df_av45.rename(columns={'visit_code': 'Visit', 'Path': 'mri_path', 'AgeatAV45Scan': 'Age'})
df_av45 = df_av45.astype({'RID': str})
df_av45['Month'] = df_av45['Visit'].apply(convert_month)

# Flute-MR
df_flute = pd.read_excel(filename, sheet_name="Flute-MR")
df_flute = df_flute[['RID', 'visit_code', 'flute_path', 'Path', 'AgeatFluteScan'] + base_columns]
df_flute = df_flute.rename(columns={'visit_code': 'Visit', 'Path': 'mri_path', 'AgeatFluteScan': 'Age'})
df_flute = df_flute.astype({'RID': str})
df_flute['Month'] = df_flute['Visit'].apply(convert_month)

# MR Only
df_mri = pd.read_excel(filename, sheet_name="MR only")
df_mri = df_mri[['Subject ID', 'VC', 'MR path', 'AgeAtMRI'] + base_columns]
df_mri = df_mri.rename(columns={'Subject ID': 'RID', 'VC': 'Visit', 'MR path': 'mri_path', 'AgeAtMRI': 'Age'})
df_mri = df_mri.astype({'RID': str})
df_mri['Month'] = df_mri['Visit'].apply(convert_month)

# Merge RID and Month
df = df_pib[['RID', 'Month']]
df = pd.merge(df, df_av45[['RID', 'Month']], how='outer')
df = pd.merge(df, df_flute[['RID', 'Month']], how='outer')
df = pd.merge(df, df_mri[['RID', 'Month']], how='outer')

df['DX'] = ''
df['Type'] = ''
flag = False

for i, row in df.iterrows():
    rid, month = row['RID'], row['Month']

    for key, temp in zip(['PIB', 'AV45', 'Flute', 'MRI'], [df_pib, df_av45, df_flute, df_mri]):
        record = temp.loc[(temp['RID'] == rid) & (temp['Month'] == month)]
        if not record.empty:
            assert not flag
            df.loc[i, 'DX'] = record['DX'].item()
            df.loc[i, 'Type'] = key
            flag = True
        else:
            flag = False


# unique patient IDs
df = df.sort_values(['RID', 'Month']).reset_index(drop=True)
patient_rids = df['RID'].unique()

# L = length of time window
for L in [18, 36]:

    new_labels = []

    for id in patient_rids:

        patient_data = df[df['RID'] == id]
        patient_data = patient_data.sort_values('Month')

        patient_labels = []
        for i, row in patient_data.iterrows():

            # MCI
            if row['DX'] == 'MCI':

                current_month = row['Month']

                # frame: current < x < current + L
                frame = patient_data[(patient_data['Month'] > current_month) &
                                     (patient_data['Month'] <= current_month + L)]
                frame_enough = (frame['Month'].max() == current_month + L)

                # future: current + L < x
                future = patient_data[patient_data['Month'] > current_month + L]

                # if frame record does not exist
                if frame.empty:
                    # if future record does not exist
                    if future.empty:
                        patient_labels.append('')
                    # if future record exists
                    else:
                        patient_labels.append('NC')
                # if frame record exists
                else:
                    if frame_enough:
                        if (frame['DX'] == 'AD').any():
                            patient_labels.append('C')
                        else:
                            patient_labels.append('NC')
                    else:
                        # if converted to AD within the time frame
                        if (frame['DX'] == 'AD').any():
                            patient_labels.append('C')
                        # not converted to AD
                        else:
                            # if future record does not exist
                            if future.empty:
                                patient_labels.append('')
                            # if future record exists
                            else:
                                patient_labels.append('NC')
            # not MCI
            else:
                patient_labels.append('')

        new_labels.extend(patient_labels)

    df[f'Conv_{L}'] = new_labels

# 2. Check number of images and patients ~ each class
for L in [18, 36]:

    print('-----------')
    # Define L
    class_label = f'Conv_{L}'

    # Count the number of rows per class, excluding empty strings
    class_counts = df[class_label].value_counts()
    class_counts = class_counts[class_counts.index != '']
    print(class_counts)

    # Count the number of unique patients in each class
    c_patients = df[df[class_label] == 'C']['RID'].nunique()
    nc_patients = df[df[class_label] == 'NC']['RID'].nunique()
    print(f"Number of unique patients in C: {c_patients}")
    print(f"Number of unique patients in NC: {nc_patients}")

    # Count the number of patients with both C and NC classes
    filtered_df = df[df[class_label] != '']
    unique_classes_per_patient = filtered_df.groupby('RID')[class_label].nunique()
    patients_with_both_classes = sum(unique_classes_per_patient > 1)
    print(f"Number of patients with both C and NC classes: {patients_with_both_classes}")


# 3. Calculate NL -> MCI conversion period
# Initialize a new column with NaN
df['Time_From_NL_to_MCI'] = np.nan

# Iterate over rows per RID
for rid, group in df.groupby('RID'):
    nl_index = None
    for idx, row in group.iterrows():
        if row['DX'] == 'NL':
            nl_index = idx
        elif row['DX'] == 'MCI' and nl_index is not None:
            df.loc[nl_index, 'Time_From_NL_to_MCI'] = row['Month'] - df.loc[nl_index, 'Month']
            nl_index = None  # Reset nl_index

# 4. save results
df.to_csv('E:/data/AIBL/AIBL_labels.csv', index=False)
