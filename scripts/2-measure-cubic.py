import os
import tqdm
import numpy as np
import glob
import nibabel as nib

DATA_DIR = "D:/data/ADNI/FS7"
brainmask_files = sorted(glob.glob(os.path.join(DATA_DIR, "*/*/brainmask.mgz")))
brain_files = sorted(glob.glob(os.path.join(DATA_DIR, "*/*/brain.mgz")))

for brainmask_file in tqdm.tqdm(brainmask_files):

    # check ids are matched
    brainmask = nib.load(brainmask_file)
    brainmask = nib.as_closest_canonical(brainmask)
    brainmask = brainmask.get_fdata()

    dims = {k: {'min': 255, 'max': 0} for k in range(3)}
    for i, axis in enumerate([(1, 2), (0, 2), (0, 1)]):
        s = np.sum(brainmask, axis=axis)
        mask = np.where(s > 0)[0]
        m, M = mask[0], mask[-1]
        if m < dims[i]['min']:
            dims[i]['min'] = m
        if M > dims[i]['max']:
            dims[i]['max'] = M
# dims
# 60 - 199 -> 139
# 43 - 210 -> 173
# 54 - 182 -> 128
###########
for k, v in dims.items():
    print(k)
    print(v['max'] - v['min'])
    print((v['max'] - v['min'])/2 + 192 / 2)
    print((v['max'] - v['min'])/2 - 192 / 2)


xmin, xmax = 30, 222
ymin, ymax = 30, 222
zmin, zmax = 30, 222

for brain_file, brainmask_file in zip(brain_files, brainmask_files):

    import matplotlib.pyplot as plt
    import seaborn as sns

    ##
    brainmask = nib.load(brainmask_file)
    brainmask = nib.as_closest_canonical(brainmask)
    brainmask = brainmask.get_fdata()
    brainmask_sliced = brainmask[xmin:xmax, ymin:ymax, zmin:zmax]

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    sns.heatmap(brainmask_sliced[96, :, :], cmap='binary', ax=axs[0], vmax=255)
    sns.heatmap(brainmask_sliced[:, 96, :], cmap='binary', ax=axs[1], vmax=255)
    sns.heatmap(brainmask_sliced[:, :, 96], cmap='binary', ax=axs[2], vmax=255)
    plt.tight_layout()
    plt.suptitle(brainmask_file)
    plt.show()

    ##
    brainmask = nib.load(brain_file)
    brainmask = nib.as_closest_canonical(brainmask)
    brainmask = brainmask.get_fdata()
    brainmask_sliced = brainmask[xmin:xmax, ymin:ymax, zmin:zmax]

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    sns.heatmap(brainmask_sliced[96, :, :], cmap='binary', ax=axs[0], vmax=255)
    sns.heatmap(brainmask_sliced[:, 96, :], cmap='binary', ax=axs[1], vmax=255)
    sns.heatmap(brainmask_sliced[:, :, 96], cmap='binary', ax=axs[2], vmax=255)
    plt.tight_layout()
    plt.suptitle(brain_file)
    plt.show()


    break
