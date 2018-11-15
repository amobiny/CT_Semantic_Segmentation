
import h5py
import numpy as np
import os
# import matplotlib.pyplot as plt
# import sys
# from sklearn.model_selection import train_test_split
# import cv2
# import skimage.transform
import glob
from scipy.ndimage import zoom


def load_resize_save(input_files_path, dest_path, new_size=128):
    """
    Get and resize images or masks
    :param ids: image ids (i.e. file names)
    :param path: path to the image folder
    :param new_size: resize images to this size
    :param array_type: type of generated arrays (bool for mask and unit8 for input image)
    :return:
    """
    for file_path in input_files_path:
        file_name = file_path.split('/')[-1]
        dest_file_path = dest_path + file_name
        if not os.path.exists(dest_file_path):
            h5f = h5py.File(file_path, 'r')
            x = np.squeeze(h5f['x'][:])
            x_norm = np.squeeze(h5f['x_norm'][:])
            y = np.squeeze(h5f['y'][:])
            h5f.close()
            print('Getting and resizing {}'.format(file_name))
            new_x = zoom(x, (0.5, 0.5, 0.5), order=5, mode='wrap')
            new_x_norm = zoom(x_norm, (0.5, 0.5, 0.5), order=5, mode='wrap')
            new_y = zoom(y, (0.5, 0.5, 0.5), order=5, mode='wrap').astype(int)
            new_y[new_y > 5] = 5
            new_y[new_y < 0] = 0
            assert new_x.shape == new_x_norm.shape == new_y.shape, 'shape does not match for '.format(file_name)
            h5f = h5py.File(dest_file_path, 'w')
            h5f.create_dataset('x', data=new_x[np.newaxis, :, :, :, np.newaxis])
            h5f.create_dataset('x_norm', data=new_x_norm[np.newaxis, :, :, :, np.newaxis])
            h5f.create_dataset('y', data=new_y[np.newaxis, :, :, :])
            h5f.close()


project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
# path of raw hdf5 scans
all_files = glob.glob(project_path + '/data_preparation/our_data/4_correctMask_normalized/*.h5')
dest_path = project_path + '/data_preparation/our_data/5_down_sampled/'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)


load_resize_save(all_files, dest_path, new_size=128)
