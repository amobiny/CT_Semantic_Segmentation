import glob
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
# path of raw hdf5 scans
all_files = glob.glob(project_path + '/data_preparation/our_data/2_hdf5_files/*.h5')
dest_path = project_path + '/data_preparation/our_data/4_correctMask_normalized/'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

all_img, all_mask, all_img_normalized = [], [], []

# load and store all files (change negative values to zero)
for file_ in all_files:
    h5f = h5py.File(file_, 'r')
    x = h5f['x'][:]
    x[x < 0] = 0
    y = h5f['y'][:]
    y[y < 0] = 0
    h5f.close()

    # correcting the mask
    y[y <= 2561] = 0
    y[y >= 6000] = 0
    y[y == 4100] = 0
    y[y == 4101] = 0
    y[y == 3000] = 1    # liver
    y[y == 3100] = 2    # spleen
    y[5000 <= y] = 4    # bone
    y[3220 <= y] = 3    # kidney
    y[2580 <= y] = 5    # vessel

    # NORMALIZE INPUT IN [0, 1]
    # x_max = np.max(x)
    # x_norm = x / (x_max+0.0)
    m = np.mean(x)
    s = np.std(x)
    x_norm = (x-m)/(s+0.0)

    # reshape and one-hot-encode the labels
    x = np.transpose(x, [1, 2, 0])[np.newaxis, :, :, :, np.newaxis]
    x_norm = np.transpose(x_norm, [1, 2, 0])[np.newaxis, :, :, :, np.newaxis]
    y = np.transpose(y, [1, 2, 0])[np.newaxis, :, :, :]

    # save
    file_name = file_.split('/')[-1]
    hdf5_file_path = dest_path + file_name
    if not os.path.exists(hdf5_file_path):  # if not already created
        h5f = h5py.File(hdf5_file_path, 'w')
        h5f.create_dataset('x', data=x)
        h5f.create_dataset('x_norm', data=x_norm)
        h5f.create_dataset('y', data=y)
        h5f.close()

    # all_img.append(x)
    # all_img_normalized.append(x_norm)
    # all_mask.append(y)

# uniqus_counts = []

# for mask in all_mask:
#     uniqus_counts.append(np.unique(mask, return_counts=True))
#
# all_counts = np.zeros((67, 6))
# for i, label_count in enumerate(uniqus_counts):
#     for label, count in zip(label_count[0], label_count[1]):
#         all_counts[i, label] = int(count)
#
# ind = np.arange(67)
# p0 = plt.bar(ind, all_counts[:, 0].T)
# p1 = plt.bar(ind, all_counts[:, 1].T)
# p2 = plt.bar(ind, all_counts[:, 2].T)
# p3 = plt.bar(ind, all_counts[:, 3].T)
# p4 = plt.bar(ind, all_counts[:, 4].T)
# p5 = plt.bar(ind, all_counts[:, 5].T)
#
# plt.legend((p0[0], p1[0], p2[0], p3[0], p4[0], p5[0]), ('background', 'liver', 'spleen', 'kidney', 'bone', 'vessel'))
# plt.show()