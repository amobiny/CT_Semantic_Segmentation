import numpy as np
import glob
import h5py

# file_names = glob.glob("/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation"
#                        "/data_preparation/our_data/4_correctMask_normalized/train/*.h5")
#
# hist = np.zeros((6))
# for file_ in file_names:
#     h5f = h5py.File(file_, 'r')
#     y = h5f['y'][:]
#     h5f.close()
#     if np.sum(y) == 0:
#         print('empty mask found in file {}'.format(file_))
#         continue
#     a, b = np.unique(y, return_counts=True)
#     if len(a) < 6:
#         print('{} doesnt contain all values; skipped for now'.format(file_))
#         continue
#     hist += b
#
# print()
#
# sum_all = np.sum(hist)
#
# print(sum_all/hist)
from utils.eval_utils import get_uncertainty_measure

h5f = h5py.File('run2_weighted_bayes.h5', 'r')
# x = h5f['x'][:]
y = h5f['y'][:]
y_var = h5f['y_var'][:]
y_pred = h5f['y_pred'][:]
h5f.close()

uncertainty_measure = get_uncertainty_measure(y, y_pred, y_var)

print(uncertainty_measure)






