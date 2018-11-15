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


def var_calculate(pred, prob_variance):
    """
    computes the uncertainty measure for a single scan
    :param pred: predicted label, shape is [image_h, image_w, image_d]
    :param prob_variance: the total variance for ALL classes, is of shape [image_h, image_w, image_d, C]
    :return: corresponding variance of shape [image_h, image_w, image_d]
    """
    pred = np.reshape(pred, [-1])
    image_h, image_w, image_d = prob_variance.shape[0], prob_variance.shape[1], prob_variance.shape[2]
    NUM_CLASS = np.shape(prob_variance)[-1]
    var_sep = []    # var_sep is the corresponding variance if this pixel choose label k
    length_cur = 0  # length_cur represent how many pixels has been read for one images
    for row in np.reshape(prob_variance, [image_h * image_w * image_d, NUM_CLASS]):
        temp = row[pred[length_cur]]
        length_cur += 1
        var_sep.append(temp)
    var_one = np.reshape(var_sep, [image_h, image_w, image_d])
    # var_one is the corresponding variance in terms of the "optimal" label

    return var_one

