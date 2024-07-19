import pandas as pd
import numpy as np

import os
import glob
import tarfile
import gzip
import tarfile
import tqdm
import shutil
import pickle
import re
from utils.logging import get_rich_pbar

def extract_tar_gz(file_path, output_path):
    with tarfile.open(file_path, 'r:gz') as file:
        file.extractall(path=output_path)


def selective_extract(file_path, target_strings, output_path='.'):

    with tarfile.open(file_path, 'r:gz') as archive:

        members = archive.getmembers()
        names = archive.getnames()

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            task = pg.add_task(f"[bold red] Start... ", total=len(members))
            desc = ''
            for member, name in zip(members, names):
                # Check if member is a directory and contains any of the target strings
                if any(s in member.name for s in target_strings):
                    archive.extract(member, path=output_path)
                    desc = name
                    pg.update(task, advance=1., description=desc)
                else:
                    pg.update(task, advance=1., description=desc)

    return members, names

# Unzip
DIR_FROM = "E:/"

MRI_DIR_TO = "E:/data/OASIS/MRI"
AV45_DIR_TO = "E:/data/OASIS/AV45"
PIB_DIR_TO = "E:/data/OASIS/PIB"

os.makedirs(MRI_DIR_TO, exist_ok=True)
os.makedirs(AV45_DIR_TO, exist_ok=True)
os.makedirs(PIB_DIR_TO, exist_ok=True)

file_parents = glob.glob(os.path.join(DIR_FROM, "*.tar.gz"))
mri_parents = [f for f in file_parents if "_Main_" in f]
av45_parents = [f for f in file_parents if "AV45" in f]
pib_parents = [f for f in file_parents if "PiB" in f]

###
# MRI using files
mri_files = os.path.join("D:/", 'data/', 'aibl_oasis_data_info_all.csv')
mri_files = pd.read_csv(mri_files)
mri_files = mri_files.loc[mri_files['Source'] == 'OASIS']
target_files = mri_files['MR Path'].tolist()
target_files = [f.replace('.pkl', '') for f in target_files]

# MRI
# unzip selected directories
for mri_parent in tqdm.tqdm(mri_parents):

    members, names = selective_extract(file_path=mri_parent,
                                       target_strings=target_files,
                                       output_path='E:/data/OASIS/MRI')

# move files and remove existing files
mri_file_names = [f for f in glob.glob(f"{MRI_DIR_TO}/**", recursive=True) if os.path.isfile(f)]

nii_names = [n for n in mri_file_names if n.endswith('.nii') and n.split('\\')[-1].startswith('wm')]
brain_names = [n for n in mri_file_names if n.endswith('\\brain.mgz')]
wmparc_names = [n for n in mri_file_names if n.endswith('\\wmparc.mgz')]
brainmask_names = [n for n in mri_file_names if n.endswith('\\brainmask.mgz')]
names = nii_names + brain_names + wmparc_names + brainmask_names

for name in names:
    name_to = name.replace(MRI_DIR_TO, 'E:/data/OASIS/MRI/temp')
    os.makedirs(os.path.join(*name_to.split('\\')[:-1]), exist_ok=True)
    shutil.move(name, name_to)
# delete existing files in E:/data/OASIS/MRI manually

###
# PiB
for fbp_parent in tqdm.tqdm(pib_parents):
    tar_parent = tarfile.TarFile.open(fbp_parent, mode='r:gz')
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

###
# AV45
for fbp_parent in tqdm.tqdm(av45_parents):
    tar_parent = tarfile.TarFile.open(fbp_parent, mode='r:gz')
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
