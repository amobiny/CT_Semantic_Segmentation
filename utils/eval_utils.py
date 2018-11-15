import numpy as np


def compute_iou(hist):
    intersection = np.diag(hist)
    ground_truth_set = hist.sum(axis=1)
    predicted_set = hist.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    acc = np.diag(hist)/np.sum(hist, axis=1)
    return IoU, acc


def get_hist(y_pred, y, num_cls):
    """
    computes the confusion matrix
    :param y_pred: flattened predictions
    :param y: flattened labels
    :param num_cls: number of classes
    :return: confusion matrix of shape (C, C)
    """
    k = (y >= 0) & (y < num_cls)
    hist = np.bincount(num_cls * y[k].astype(int) + y_pred[k], minlength=num_cls ** 2).reshape(num_cls, num_cls)
    return hist




