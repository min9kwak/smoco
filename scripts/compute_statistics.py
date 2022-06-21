import os
import pickle
import pandas as pd
import tqdm

root = 'D:/data/ADNI'
data_info = 'labels/data_info.csv'
data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str, 'Conv': int})
data_info = data_info.loc[data_info.IS_FILE]

mri_files = [os.path.join(root, 'template/FS7', f'{i}.pkl') for i in data_info.MRI]

data_info = data_info.loc[~data_info.PET.isna()]
pet_files = [os.path.join(root, 'template/PUP_FBP', f'{i}.pkl') for i in data_info.PET]

# mri
m1, M1 = +1000, -1000
for mri_file in tqdm.tqdm(mri_files):
    with open(mri_file, 'rb') as fb:
        img = pickle.load(fb)
    m1_, M1_ = img.min(), img.max()
    if m1_ < m1:
        m1 = m1_
    if M1_ > M1:
        M1 = M1_

m2, M2 = +1000, -1000
neg_count = 0
for pet_file in tqdm.tqdm(pet_files):
    with open(pet_file, 'rb') as fb:
        img = pickle.load(fb)
    m2_, M2_ = img.min(), img.max()
    if m2_ < m2:
        m2 = m2_
    if M2_ > M2:
        M2 = M2_
    if m2_ < 0:
        neg_count += 1
print(m1, M1)
print(m2, M2)

result = {'mri': {'min': m1, 'max': M1},
          'pet': {'min': m2, 'max': M2}}

with open(os.path.join(root, 'labels/minmax.pkl'), 'wb') as fb:
    pickle.dump(result, fb)