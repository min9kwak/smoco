import os
import glob
import tarfile
import gzip
import tqdm
import shutil
import pickle

# define directory
MRI_DIR_FROM = "D:/data/ADNI/source files/FS7"
MRI_DIR_TO = "D:/data/ADNI/FS7/"
os.makedirs(MRI_DIR_TO, exist_ok=True)

# 1. MRI - nu.mgz
mri_parents = glob.glob(os.path.join(MRI_DIR_FROM, "*.tar"))
logs = []
for mri_parent in tqdm.tqdm(mri_parents):
    tar_parent = tarfile.TarFile(mri_parent)
    tar_parent_names = tar_parent.getnames()

    path_parent = mri_parent.replace(".tar", "")
    tar_parent.extractall(path=path_parent)

    for tar_parent_name in tqdm.tqdm(tar_parent_names):
        tar_child = tarfile.open(os.path.join(path_parent, tar_parent_name))
        tar_child_names = tar_child.getnames()

        nu_names = [n for n in tar_child_names if n.endswith('/nu.mgz')]
        tar_child_names = nu_names
        for tar_child_name in tar_child_names:
            tar_child.extract(tar_child_name, path=os.path.join(MRI_DIR_TO))
        del tar_child

    shutil.rmtree(path_parent, ignore_errors=True)

# 2. Cubic size of brainmask
import pandas as pd
import nibabel as nib
import numpy as np

DATA_DIR = "D:/data/ADNI"
data_info = pd.read_csv('D:/data/ADNI/labels/data_info.csv')
data_info = data_info.loc[data_info.IS_FILE]
# with open(os.path.join(DATA_DIR, 'labels/mri_abnormal.pkl'), 'rb') as fb:
#     mri_abnormal = pickle.load(fb)
# data_info = data_info.loc[~data_info.MRI.isin(mri_abnormal)]

# MRI brain - whole
brainmask_files = sorted([os.path.join(DATA_DIR, 'FS7', row.MRI, 'mri/brainmask.mgz') for _, row in data_info.iterrows()])
brain_dims = {k: {'min': 255, 'max': 0, 'len': 0} for k in range(3)}

for brainmask_file in tqdm.tqdm(brainmask_files):

    # check ids are matched
    brainmask = nib.load(brainmask_file)
    brainmask = nib.as_closest_canonical(brainmask)
    brainmask = brainmask.get_fdata()

    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(brainmask, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        l = M - m
        if m < brain_dims[i]['min']:
            brain_dims[i]['min'] = m
        if M > brain_dims[i]['max']:
            brain_dims[i]['max'] = M
        if l > brain_dims[i]['len']:
            brain_dims[i]['len'] = l

# 49 / 210 / 161 (161) -- 29 / 236 / 198 (207) -- 23 / 197 / 159 (174) --> 210
m, M = 128 - 105, 128 + 105

SAVE_DIR = 'D:/data/ADNI/individual/FS7/'
os.makedirs(SAVE_DIR, exist_ok=True)
for brainmask_file in tqdm.tqdm(brainmask_files):

    # brainmask
    brainmask = nib.load(brainmask_file)
    brainmask = nib.as_closest_canonical(brainmask)
    brainmask = brainmask.get_fdata()
    brainmask = brainmask[m:M, m:M, m:M] # 210, 210, 210

    # nu.mgz --> masking
    nu_file = brainmask_file.replace('brainmask.mgz', 'nu.mgz')
    nu = nib.load(nu_file)
    nu = nib.as_closest_canonical(nu)
    nu = nu.get_fdata()
    nu = nu[m:M, m:M, m:M] # 210, 210, 210

    mask = (brainmask == 0)

    from skimage.transform import resize
    mask = resize(mask, [145, 145, 145])
    nu = resize(nu, [145, 145, 145])
    nu[mask] = 0.0

    id = brainmask_file.split('\\')[2]
    with open(os.path.join(SAVE_DIR, f'{id}.pkl'), 'wb') as fb:
        pickle.dump(nu, fb)
