import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, auc
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import h5py


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


def get_uncertainty_precision(y, y_pred, y_var):
    """
    computes the precision of the uncertainty map
    :param y:input label of shape [#images, height, width]
    :param y_pred: predicted masks of shape [#images, height, width]
    :param y_var: uncertainty maps of shape [#images, height, width]
    :return: precision of the uncertainty map generated by the network;
            the larger this value, the better the uncertainty map
    """
    # first let's normalize the uncertainty maps so that its values sum to one for each input image
    # It allows us to get a final precision value which is in range [0, 1]
    norm_factor = np.sum(y_var, axis=(1, 2))   # sum of uncertainty values for each image
    y_var /= norm_factor[:, np.newaxis, np.newaxis]

    wrong_pred = (y != y_pred).astype(int)
    true_pos = np.sum(wrong_pred * y_var)      # we want this to be high

    correct_pred = (y == y_pred).astype(int)
    false_pos = np.sum(correct_pred * y_var)      # we want this to be low

    precision = true_pos / (true_pos + false_pos)
    return precision


def predictive_entropy(mean_prob):
    eps = 1e-5
    return -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=-1)


def mutual_info(mean_prob, mc_prob):
    """
    computes the mutual information
    :param mean_prob: average MC probabilities of shape [batch_size, img_h, img_w, num_cls]
    :param mc_prob: List MC probabilities of length mc_simulations;
                    each of shape  of shape [batch_size, img_h, img_w, num_cls]
    :return: mutual information of shape [batch_size, img_h, img_w, num_cls]
    """
    eps = 1e-5
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=-1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=-1)
    return first_term + second_term


def plot_precision_recall_curve(y, y_pred, y_var):
    # norm_factor = np.sum(y_var, axis=(1, 2))   # sum of uncertainty values for each image
    # y_var /= norm_factor[:, np.newaxis, np.newaxis]

    wrong_pred = (y != y_pred).astype(int)
    precision, recall, _ = precision_recall_curve(wrong_pred.reshape([-1]), y_var.reshape([-1]))
    average_precision = average_precision_score(wrong_pred.reshape([-1]), y_var.reshape([-1]))
    return precision, recall, average_precision


def compute_metrics(run_name, num_split=20):
    h5f = h5py.File(run_name + '.h5', 'r')
    y = h5f['y'][:]
    y_pred = h5f['y_pred'][:]
    y_var = h5f['y_var'][:]
    h5f.close()
    umin = np.min(y_var)
    umax = np.max(y_var)
    N_tot = np.prod(y.shape)
    wrong_pred = (y != y_pred).astype(int)

    precision_, recall_, threshold = precision_recall_curve(wrong_pred.reshape([-1]), y_var.reshape([-1]))

    # TODO: what is the best way to pick thresholds?!
    uT = np.linspace(umin, umax, num_split)
    # uT = np.append(threshold[::100000], threshold[-1])

    right_pred = (y == y_pred).astype(int)
    npv, recall, acc, precision, T = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    counter = 0
    for ut in uT:
        t = (ut - umin) / (umax - umin)
        counter += 1
        uncertain = (y_var >= ut).astype(int)
        certain = (y_var < ut).astype(int)
        TP = np.sum(uncertain * wrong_pred)
        TN = np.sum(certain * right_pred)
        N_w = np.sum(wrong_pred)
        N_c = np.sum(certain)
        N_unc = np.sum(uncertain)
        recall = np.append(recall, TP / N_w)
        npv = np.append(npv, TN / N_c)
        precision = np.append(precision, TP/N_unc)
        acc = np.append(acc, (TN + TP) / N_tot)
        T = np.append(T, t)
    auc_ = auc(recall_, precision_)
    return recall, npv, acc, precision_, recall_, auc_, T
