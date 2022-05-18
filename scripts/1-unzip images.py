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

FBP_DIR_FROM = "D:/data/ADNI/source files/PUP_FBP/"
# FBP_DIR_TO = "D:/data/ADNI/PUP_FBP/"
FBP_DIR_TO = "D:/data/ADNI/PUP_FBP_/"

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

        # brain_names = [n for n in tar_child_names if n.endswith('/brain.mgz')]
        # wmparc_names = [n for n in tar_child_names if n.endswith('/wmparc.mgz')]
        # if len(wmparc_names) == 0:
        #     missed = {mri_parent: tar_parent_name}
        #     print(missed)
        #     logs.append(missed)
        # tar_child_names = brain_names + wmparc_names
        brainmask_names = [n for n in tar_child_names if n.endswith('/brainmask.mgz')]
        tar_child_names = brainmask_names
        for tar_child_name in tar_child_names:
            tar_child.extract(tar_child_name, path=os.path.join(MRI_DIR_TO))
        del tar_child
    shutil.rmtree(path_parent, ignore_errors=True)

with open("logs.pkl", 'wb') as f:
    pickle.dump(logs, f)

# FBP
fbp_parents = glob.glob(os.path.join(FBP_DIR_FROM, "*.tar"))
for fbp_parent in tqdm.tqdm(fbp_parents):
    tar_parent = tarfile.TarFile(fbp_parent)
    tar_parent_names = tar_parent.getnames()

    path_parent = fbp_parent.replace(".tar", "")
    tar_parent.extractall(path=path_parent)

    for tar_parent_name in tqdm.tqdm(tar_parent_names):
        tar_child = tarfile.open(os.path.join(path_parent, tar_parent_name))
        tar_child_names = tar_child.getnames()

        param_names = [n for n in tar_child_names if ("pet_proc" in n) and (n.endswith("param"))]
        nii_names = [n for n in tar_child_names if n.endswith(".nii.gz")]

        names = param_names + nii_names
        for name in names:
            tar_child.extract(name, path=FBP_DIR_TO)
    shutil.rmtree(path_parent, ignore_errors=True)

# FBP - nii
nii_names = glob.glob(os.path.join(FBP_DIR_TO, "*/*/*.nii.gz"))
for nii_name in tqdm.tqdm(nii_names):
    with gzip.open(nii_name, 'rb') as f_in:
        with open(nii_name.replace('.gz', ''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(nii_name)
