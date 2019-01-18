import glob
import numpy as np
import time
import h5py
import os
project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
all_files = glob.glob(project_path + '/data_preparation/our_data/1_raw_data/*.segdat')
destination_path = project_path + '/data_preparation/our_data/2_hdf5_files/'
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

img_counter = 0

for file_path in all_files:
    start_time = time.time()
    name_splitted = file_path.split('/')[-1].split('.')
    if len(name_splitted) > 3:
        file_name = name_splitted[0] + '_' + name_splitted[1] + '_' + name_splitted[2] + '_' + name_splitted[-3]
    else:
        file_name = file_path.split('/')[-1].split('.')[0]
    hdf5_file_path = destination_path + file_name + '.h5'
    empty_file_path = project_path + '/data_preparation/our_data/3_empty_files/' + file_name + '.h5'
    if os.path.exists(hdf5_file_path) or os.path.exists(empty_file_path):
        continue
    with open(file_path, 'r') as content_file:
        img_counter += 1
        print('-' * 50)
        print('reading file #{}: {}'.format(img_counter, file_name))
        content = content_file.read()
        bad_vals = [str(bad_val) for bad_val in range(-1, -10, -1)]
        raw_data = np.array([int(dig) for dig in content.split() if dig.isdigit() or dig in bad_vals])
        num_slices = raw_data[0]
        data = raw_data[-num_slices*2*512*512:]     # take the entries from the end to avoid the initial wrong values
        data[data < 0] = 0  # remove negative values
        image = data[0:len(data):2].reshape(num_slices, 512, 512)
        mask = data[1:len(data):2].reshape(num_slices, 512, 512)
        run_time = time.time() - start_time
        np.unique([mask])
        np.unique([mask]).shape
    print('run-time for image #{} with {} slices was: {} seconds'.format(img_counter, num_slices, int(run_time)))
    print('saving ...')
    h5f = h5py.File(hdf5_file_path, 'w')
    h5f.create_dataset('x', data=image)
    h5f.create_dataset('y', data=mask)
    h5f.create_dataset('slice_count', data=num_slices)
    h5f.close()
