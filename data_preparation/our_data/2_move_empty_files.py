import glob
import h5py
import numpy as np
import os

project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
# path of raw hdf5 scans
all_files = glob.glob(project_path + '/data_preparation/our_data/2_hdf5_files/*.h5')
# path to move empty or corrupted files into it
empty_file_dest_path = (project_path + '/data_preparation/our_data/3_empty_files/')
if not os.path.exists(empty_file_dest_path):
    os.makedirs(empty_file_dest_path)

empty_img, empty_mask, empty_file = [], [], []
for file_ in all_files:
    h5f = h5py.File(file_, 'r')
    x = h5f['x'][:]
    y = h5f['y'][:]
    h5f.close()
    if not np.sum(x) and not np.sum(y):
        empty_file += [file_]
    elif np.sum(x) == 0:
        empty_img += [file_]
    elif np.sum(y) == 0:
        empty_mask += [file_]


def move_empty(file_current_path):
    global empty_file_dest_path
    file_name = file_current_path.split('/')[-1]
    # move the file
    os.rename(file_current_path, empty_file_dest_path+file_name)


map(move_empty, empty_file)
map(move_empty, empty_img)
map(move_empty, empty_mask)



