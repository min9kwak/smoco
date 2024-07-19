import os
import glob
import tarfile
import gzip

import pandas as pd
import tqdm
import shutil
import pickle

# define directory
MRI_DIR_FROM = "D:/data/ADNI/source files/FS7"
MRI_DIR_TO = "D:/data/ADNI/FS7/"

FBP_DIR_FROM = "D:/data/ADNI/source files/PUP_FBP/"
FBP_DIR_TO = "D:/data/ADNI/PUP_FBP/"

os.makedirs(MRI_DIR_TO, exist_ok=True)
os.makedirs(FBP_DIR_TO, exist_ok=True)

# MRI
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
        wmm_names = [n for n in tar_child_names if 'cat12/mri/wmm' in n]
        tar_child_names = wmm_names
        for tar_child_name in tar_child_names:
            tar_child.extract(tar_child_name, path=os.path.join(MRI_DIR_TO))
        del tar_child
    shutil.rmtree(path_parent, ignore_errors=True)
