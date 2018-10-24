import glob
import h5py
import numpy as np
import os

project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
# path of raw hdf5 scans
all_files = glob.glob(project_path + '/data_preparation/our_data/2_hdf5_files/*.h5')

all_img, all_mask = [], []

# load and store all files (change negative values to zero)
for file_ in all_files:
    h5f = h5py.File(file_, 'r')
    x = h5f['x'][:]
    x[x < 0] = 0
    y = h5f['y'][:]
    y[y < 0] = 0
    h5f.close()
    all_img.append(x)
    all_mask.append(y)

uniqus_counts = []

for mask in all_mask:
    uniqus_counts.append(np.unique(mask, return_counts=True))

uniq_vals = []
for mask in all_mask:
    uniq_vals += list(np.unique(mask))
# uniq_vals = set(uniq_vals)


print()