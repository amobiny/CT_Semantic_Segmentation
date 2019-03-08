import h5py
import numpy as np
from utils.eval_utils import get_hist

run_name = 'dropconnect_MI_uncertainty'
num_cls = 6


def compute_metrics(hist):
    intersection = np.diag(hist)
    ground_truth_set = hist.sum(axis=1)
    predicted_set = hist.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    mIoU = np.mean(intersection / union.astype(np.float32))
    p_acc = np.sum(np.diag(hist))/np.sum(hist)
    m_acc = (1/hist.shape[0]) * np.sum(np.diag(hist) / np.sum(hist, axis=1))
    return mIoU, p_acc, m_acc


h5f = h5py.File(run_name + '.h5', 'r')
y = h5f['y'][:]
y_pred = h5f['y_pred'][:]
h5f.close()

hist = get_hist(y_pred.astype(int), y.astype(int), num_cls)
mean_iou, pixel_acc, mean_acc = compute_metrics(hist)


print()




