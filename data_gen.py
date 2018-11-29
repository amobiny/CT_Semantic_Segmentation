import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob


file_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/data_preparation/our_data/6_2d/new_test_2d.h5'
h5f = h5py.File(file_path, 'r')
x_norm_1 = h5f['x_norm'][:]
y_1 = h5f['y'][:]
h5f.close()

file_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/data_preparation/our_data/6_2d/test_2d.h5'
h5f = h5py.File(file_path, 'r')
x_norm_2 = h5f['x_norm'][:]
y_2 = h5f['y'][:]
h5f.close()

x = np.concatenate((x_norm_1, x_norm_2), axis=0)
y = np.concatenate((y_1, y_2), axis=0)


dest_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/test_2d.h5'
h5f = h5py.File(dest_path, 'w')
h5f.create_dataset('x_norm', data=x)
h5f.create_dataset('y', data=y)
h5f.close()

print()

