import tensorflow as tf
import tensorlayer as tl
from sklearn.metrics import confusion_matrix
import numpy as np


def get_num_channels(x):
    """
    returns the input's number of channels
    :param x: input tensor with shape [batch_size, ..., num_channels]
    :return: number of channels
    """
    return x.get_shape().as_list()[-1]


def cross_entropy(y, logits, n_class):
    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(y, [-1, n_class])
    try:
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=flat_logits, labels=flat_labels))
    except:
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=flat_logits, labels=flat_labels))
    return loss


def dice_coeff(y, logits):
    # eps = 1e-5
    # prediction = pixel_wise_softmax(logits)
    # intersection = tf.reduce_sum(prediction * y)
    # union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    # dice_loss = 1 - (2 * intersection / union)
    outputs = tl.act.pixel_wise_softmax(logits)
    dice_loss = 1 - tl.cost.dice_coe(outputs, y, loss_type='jaccard', axis=(1, 2, 3, 4))
    return dice_loss


def pixel_wise_softmax(output_map):
    num_classes = output_map.get_shape().as_list()[-1]
    exponential_map = tf.exp(output_map)
    try:
        sum_exp = tf.reduce_sum(exponential_map, 4, keepdims=True)
    except:
        sum_exp = tf.reduce_sum(exponential_map, 4, keep_dims=True)
    # tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    tensor_sum_exp = tf.tile(sum_exp, (1, 1, 1, 1, num_classes))
    return tf.div(exponential_map, tensor_sum_exp)


def add_noise(batch, mean=0, var=0.1, amount=0.01, mode='pepper'):
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)


def write_spec(args):
    config_file = open(args.modeldir + args.run_name + '/config.txt', 'w')
    config_file.write('model: ' + args.run_name + '\n')
    config_file.write('num_cls: ' + str(args.num_cls) + '\n')
    config_file.write('optimizer: ' + 'Adam' + '\n')
    config_file.write('    learning_rate: ' + str(args.init_lr) + ' : ' + str(args.lr_min) + '\n')
    config_file.write('loss_type: ' + args.loss_type + '\n')
    config_file.write('batch_size: ' + str(args.batch_size) + '\n')
    config_file.write('data_augmentation: ' + str(args.data_augment) + '\n')
    config_file.write('    max_angle: ' + str(args.max_angle) + '\n')
    config_file.write('num_training: ' + str(args.num_tr) + '\n')
    config_file.write('keep_prob: ' + str(args.keep_prob) + '\n')
    config_file.write('batch_normalization: ' + str(args.use_BN) + '\n')
    config_file.write('kernel_size: ' + str(args.filter_size) + '\n')
    config_file.close()


def compute_iou(y_pred, y_label, num_cls):
    y_pred = y_pred.flatten()
    y_label = y_label.flatten()
    current = confusion_matrix(y_label, y_pred, labels=list(range(num_cls)))
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return IoU