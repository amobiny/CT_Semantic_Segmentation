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


def var_calculate_2d(pred, prob_variance):
    """
    computes the uncertainty measure for a single image
    :param pred: predicted label, shape is [#images, image_h, image_w]
    :param prob_variance: the total variance for ALL classes wrt each pixel,
                        is of shape [#images, image_h, image_w, C]
    :return: corresponding variance of shape [#images, image_h, image_w]
    """
    image_h, image_w = prob_variance.shape[1], prob_variance.shape[2]
    NUM_CLASS = np.shape(prob_variance)[-1]
    var_one = np.zeros((0, image_h, image_w))
    for ii in range(pred.shape[0]):
        pred_ii = np.reshape(pred[ii], [-1])
        var_sep = []    # var_sep is the corresponding variance if this pixel choose label k
        length_cur = 0  # length_cur represent how many pixels has been read for one image
        for row in np.reshape(prob_variance[ii], [image_h * image_w, NUM_CLASS]):
            temp = row[pred_ii[length_cur]]
            length_cur += 1
            var_sep.append(temp)
        var_o = np.reshape(var_sep, [1, image_h, image_w])
        # var_o is the corresponding variance in terms of the "optimal" label

        var_one = np.concatenate((var_one, var_o), axis=0)
    return var_one


def var_calculate_3d(pred, prob_variance):
    """
    computes the uncertainty measure for a single image
    :param pred: predicted label, shape is [image_h, image_w, image_d]
    :param prob_variance: the total variance for ALL classes, is of shape [image_h, image_w, image_d, C]
    :return: corresponding variance of shape [image_h, image_w, image_d]
    """
    pred = np.reshape(pred, [-1])
    image_h, image_w, image_d = prob_variance.shape[0], prob_variance.shape[1], prob_variance.shape[2]
    NUM_CLASS = np.shape(prob_variance)[-1]
    var_sep = []    # var_sep is the corresponding variance if this pixel choose label k
    length_cur = 0  # length_cur represent how many pixels has been read for one image
    for row in np.reshape(prob_variance, [image_h * image_w * image_d, NUM_CLASS]):
        temp = row[pred[length_cur]]
        length_cur += 1
        var_sep.append(temp)
    var_one = np.reshape(var_sep, [image_h, image_w, image_d])
    # var_one is the corresponding variance in terms of the "optimal" label
    return var_one


def get_uncertainty_measure(x, y, y_pred, y_var):
    """
    computes the uncertainty measure
    :param x: input images of shape [#images, height, width, channels]
    :param y:input label of shape [#images, height, width]
    :param y_pred: predicted masks of shape [#images, height, width]
    :param y_var: uncertainty maps of shape [#images, height, width]
    :return: network uncertainty measure; the bigger this metric is, the better the network is
    """
    wrong_pred = (y != y_pred).astype(int)
    wrong_unc = np.sum(wrong_pred * y_var)      # we want this to be high

    correct_pred = (y == y_pred).astype(int)
    correct_unc = np.sum(correct_pred * y_var)      # we want this to be low

    uncertain_metric = wrong_unc / correct_unc
    return uncertain_metric
