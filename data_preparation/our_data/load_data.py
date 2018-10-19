import glob
import numpy as np
import time
from tqdm import *
import matplotlib.pyplot as plt
import h5py
import cPickle as pickle

all_files = glob.glob('/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/data_preparation/our_data/raw_data/*.segdat')

# h5f = h5py.File('file_name.h5', 'r')
# x = h5f['x'][:]
# y = h5f['y'][:]
# scan_name = h5f['scan_name'][:]
# slice_count = h5f['slice_count'][:]
# h5f.close()


data_dict = {}
img_counter = 0
max_num_slice = 0

for file_path in all_files[:5]:
    img_counter += 1
    start_time = time.time()
    print('-'*50)
    f = open(file_path, "r")
    raw_text = f.readlines()
    num_slices = len(raw_text)-2
    if max_num_slice < num_slices:
        max_num_slice = num_slices
    img = np.zeros((512 * 512, num_slices))
    mask = np.zeros((512 * 512, num_slices))
    for i in tqdm(range(num_slices)):
        bad_vals = [str(i) for i in range(-1, -10, -1)]
        data = np.array([int(dig) for dig in raw_text[i+2].split() if dig.isdigit() or dig in bad_vals])
        img[:, i] = data[0:len(data):2]
        mask[:, i] = data[1:len(data)+1:2]
    run_time = time.time() - start_time
    data_dict[file_path.split('/')[-1].split('.')[0]] = \
        (img.reshape(512, 512, num_slices), mask.reshape(512, 512, num_slices))
    print('run-time for image #{} with {} slices was: {} seconds'.format(img_counter, num_slices, int(run_time)))

file_name = np.array(())
image = np.zeros((len(all_files), 512, 512, max_num_slice))
mask = np.zeros((len(all_files), 512, 512, max_num_slice))
slice_count = np.array(())
for i, scan in enumerate(data_dict.items()):
    scan_name, vals = scan
    img, msk = vals[0], vals[1]
    file_name = np.append(file_name, scan_name)
    num_slice = img.shape[-1]
    image[i, :, :, :num_slice] = img
    mask[i, :, :, :num_slice] = msk
    slice_count = np.append(slice_count, int(num_slice))


h5f = h5py.File('ra.h5', 'w')
h5f.create_dataset('x', data=image)
h5f.create_dataset('y', data=mask)
h5f.create_dataset('scan_name', data=file_name)
h5f.create_dataset('slice_count', data=slice_count)
h5f.close()






# with open('organ_data.pickle', 'w') as f:
#     pickle.dump(data_dict, f)
#
# with open('organ_data.pickle', 'rb') as f:
#     data_dict = pickle.load(f)
#

# plot the number of slices
# y = np.array([(name, dat[1].shape[-1]) for name, dat in data_dict.items()])
# plt.bar(range(1, len(y[:, -1])+1), y[:, -1].astype(int))
# plt.xlabel('scan number', fontsize=18)
# plt.ylabel('number of slices', fontsize=18)
#
# # plot the values in all the scans
# counts = [(name, np.unique(dat[1], return_counts=True)) for name, dat in data_dict.items()]
#
# labels = [0, 2584, 3000, 3100, 3221, 3231]

