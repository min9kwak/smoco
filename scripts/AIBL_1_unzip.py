import pandas as pd
import numpy as np

import os
import glob
import tarfile
import gzip
import tqdm
import shutil
import pickle
import re


# Unzip
DIR_FROM = "E:/"

MRI_DIR_TO = "E:/data/AIBL/MRI"
AV45_DIR_TO = "E:/data/AIBL/AV45"
PIB_DIR_TO = "E:/data/AIBL/PIB"

os.makedirs(MRI_DIR_TO, exist_ok=True)
os.makedirs(AV45_DIR_TO, exist_ok=True)
os.makedirs(PIB_DIR_TO, exist_ok=True)

file_parents = glob.glob(os.path.join(DIR_FROM, "*.tar"))
mri_parents = [f for f in file_parents if "_PUP_" not in f]
av45_parents = [f for f in file_parents if "AV45" in f]
pib_parents = [f for f in file_parents if "PiB" in f]

###
# MRI
logs = []
for mri_parent in tqdm.tqdm(mri_parents):

    tar_parent = tarfile.TarFile(mri_parent)
    tar_parent_names = tar_parent.getnames()

    path_parent = mri_parent.replace(".tar", "")
    tar_parent.extractall(path=path_parent)

    nii_names = [n for n in tar_parent_names if n.endswith('.nii') and n.split('/')[-1].startswith('wm')]
    brain_names = [n for n in tar_parent_names if n.endswith('/brain.mgz')]
    wmparc_names = [n for n in tar_parent_names if n.endswith('/wmparc.mgz')]
    brainmask_names = [n for n in tar_parent_names if n.endswith('/brainmask.mgz')]

    names = nii_names + brain_names + wmparc_names + brainmask_names

    for name in names:
        name_from = os.path.join(path_parent, name)
        name_to = os.path.join(MRI_DIR_TO, name)
        os.makedirs(os.path.join(*name_to.split('/')[:-1]), exist_ok=True)
        shutil.move(name_from, name_to)
    shutil.rmtree(path_parent, ignore_errors=True)

###
# AV45
for fbp_parent in tqdm.tqdm(av45_parents):
    tar_parent = tarfile.TarFile(fbp_parent)
    tar_parent_names = tar_parent.getnames()

    path_parent = fbp_parent.replace(".tar", "")
    tar_parent.extractall(path=path_parent)

    param_names = [n for n in tar_parent_names if ("pet_proc" in n) and (n.endswith("param"))]
    nii_names = [n for n in tar_parent_names if n.endswith(".nii.gz")]
    names = param_names + nii_names

    for name in tqdm.tqdm(names):
        name_from = os.path.join(path_parent, name)
        name_to = os.path.join(AV45_DIR_TO, name)
        os.makedirs(os.path.join(*name_to.split('/')[:-1]), exist_ok=True)
        shutil.move(name_from, name_to)
    shutil.rmtree(path_parent, ignore_errors=True)

# AV45 - nii
nii_names = glob.glob(os.path.join(AV45_DIR_TO, "*/*/*.nii.gz"))
for nii_name in tqdm.tqdm(nii_names):
    with gzip.open(nii_name, 'rb') as f_in:
        with open(nii_name.replace('.gz', ''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(nii_name)

###
# PiB
for fbp_parent in tqdm.tqdm(pib_parents):
    tar_parent = tarfile.TarFile(fbp_parent)
    tar_parent_names = tar_parent.getnames()

    path_parent = fbp_parent.replace(".tar", "")
    tar_parent.extractall(path=path_parent)

    param_names = [n for n in tar_parent_names if ("pet_proc" in n) and (n.endswith("param"))]
    nii_names = [n for n in tar_parent_names if n.endswith(".nii.gz")]
    names = param_names + nii_names

    for name in tqdm.tqdm(names):
        name_from = os.path.join(path_parent, name)
        name_to = os.path.join(PIB_DIR_TO, name)
        os.makedirs(os.path.join(*name_to.split('/')[:-1]), exist_ok=True)
        shutil.move(name_from, name_to)
    shutil.rmtree(path_parent, ignore_errors=True)

# PiB - nii
nii_names = glob.glob(os.path.join(PIB_DIR_TO, "*/*/*.nii.gz"))
for nii_name in tqdm.tqdm(nii_names):
    with gzip.open(nii_name, 'rb') as f_in:
        with open(nii_name.replace('.gz', ''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(nii_name)
