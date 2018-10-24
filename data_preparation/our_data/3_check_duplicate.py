import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

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


def check_duplicate(array_list):
    for i, array_1 in enumerate(array_list):
        for j, array_2 in enumerate(array_list[i+1:]):
            if array_1.shape[0] == array_2.shape[0]:
                if np.sum(array_1 - array_2) == 0:
                    print('duplicate found')


check_duplicate(all_img)
check_duplicate(all_mask)

# plot the number of slices
num_slices = []
for image in all_img:
    num_slices.append(image.shape[0])
plt.bar(range(1, len(num_slices)+1), num_slices)
plt.xlabel('scan number', fontsize=18)
plt.ylabel('number of slices', fontsize=18)
plt.show()
