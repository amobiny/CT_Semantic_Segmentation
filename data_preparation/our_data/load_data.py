import glob
import numpy as np
import time
from tqdm import *
import matplotlib.pyplot as plt
import pickle

all_files = glob.glob('/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/data_preparation/our_data/raw_data/*.segdat')

data_dict = {}
img_counter = 0
for file_path in all_files:
    img_counter += 1
    start_time = time.time()
    print('-'*50)
    f = open(file_path, "r")
    raw_text = f.readlines()
    num_slices = len(raw_text)-2
    img = np.zeros((512 * 512, num_slices))
    mask = np.zeros((512 * 512, num_slices))
    for i in tqdm(range(num_slices)):
        data = np.array([int(dig) for dig in raw_text[i+2].split() if dig.isdigit()])
        img[:, i] = data[0:len(data):2]
        mask[:, i] = data[1:len(data):2]
    run_time = time.time() - start_time
    data_dict[file_path.split('/')[-1].split('.')[0]] = \
        (img.reshape(512, 512, num_slices), mask.reshape(512, 512, num_slices))
    print('run-time for image #{} with {} slices was: {} seconds'.format(img_counter, num_slices, int(run_time)))

with open('organ_data.pickle', 'w') as f:
    pickle.dump(data_dict, f)

# with open('organ_data.pickle') as f:
#     data_dict = pickle.load(f)


# plot the number of slices
y = np.array([(name, dat[1].shape[-1]) for name, dat in data_dict.items()])
plt.bar(range(1, len(y[:, -1])+1), y[:, -1].astype(int))
plt.xlabel('scan number', fontsize=18)
plt.ylabel('number of slices', fontsize=18)

# plot the values in all the scans
counts = [(name, np.unique(dat[1], return_counts=True)) for name, dat in data_dict.items()]

labels = [0, 2584, 3000, 3100, 3221, 3231]

# for scan_name, scan_uniqs in counts:
#     scan_labels, label_counts = scan_uniqs
#     print(scan_labels, label_counts)
#     for label, count in scan_uniqs:
