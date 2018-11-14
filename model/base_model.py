import tensorflow as tf
from Data_Loader import DataLoader
from plot_utils import plot_save_preds
from utils import cross_entropy, dice_coeff, compute_iou, weighted_cross_entropy, get_hist
import os
import numpy as np
import time


class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.bayes = conf.bayes
        self.input_shape = [None, None, None, None, self.conf.channel]
        self.output_shape = [None, None, None, None]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.inputs_pl = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.labels_pl = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
            self.keep_prob_pl = tf.placeholder(tf.float32)

    def loss_func(self):
        with tf.name_scope('Loss'):
            y_one_hot = tf.one_hot(self.labels_pl, depth=self.conf.num_cls, axis=4, name='y_one_hot')
            if self.conf.weighted_loss:
                loss = weighted_cross_entropy(y_one_hot, self.logits, self.conf.num_cls)
            else:
                if self.conf.loss_type == 'cross-entropy':
                    with tf.name_scope('cross_entropy'):
                        loss = cross_entropy(y_one_hot, self.logits, self.conf.num_cls)
                elif self.conf.loss_type == 'dice':
                    with tf.name_scope('dice_coefficient'):
                        loss = dice_coeff(y_one_hot, self.logits)
            with tf.name_scope('total'):
                if self.conf.use_reg:
                    with tf.name_scope('L2_loss'):
                        l2_loss = tf.reduce_sum(
                            self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
                        self.total_loss = loss + l2_loss
                else:
                    self.total_loss = loss
                self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            self.y_pred = tf.argmax(self.logits, axis=4, name='decode_pred')
            correct_prediction = tf.equal(self.labels_pl, self.y_pred, name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=1000,
                                                   decay_rate=0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        if self.conf.random_crop:
            slice = int(self.conf.crop_size[-1]/2)
        else:
            slice = int(self.conf.depth/2)
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss),
                        tf.summary.scalar('accuracy', self.mean_accuracy),
                        tf.summary.image('train/original_image',
                                         self.inputs_pl[:, :, :, slice, :],
                                         max_outputs=5),
                        tf.summary.image('train/prediction_mask',
                                         tf.cast(tf.expand_dims(self.y_pred[:, :, :, slice], -1),
                                                 tf.float32),
                                         max_outputs=5),
                        tf.summary.image('train/original_mask',
                                         tf.cast(tf.expand_dims(self.labels_pl[:, :, :, slice], -1), tf.float32),
                                         max_outputs=5)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, is_train):
        # print('----> Summarizing at step {}'.format(step))
        if is_train:
            self.train_writer.add_summary(summary, step)
        else:
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_accuracy = 0
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
        else:
            print('----> Start Training')
        self.data_reader = DataLoader(self.conf)
        self.numValid = self.data_reader.count_num_samples(mode='valid')
        self.num_val_batch = int(self.numValid / self.conf.val_batch_size)
        for train_step in range(self.conf.max_step + 1):
            x_batch, y_batch = self.data_reader.next_batch(mode='train')
            feed_dict = {self.inputs_pl: x_batch,
                         self.labels_pl: y_batch,
                         self.is_training_pl: True,
                         self.with_dropout_pl: True,
                         self.keep_prob_pl: 0.5}
            if train_step % self.conf.SUMMARY_FREQ == 0:
                _, _, _, summary = self.sess.run([self.train_op,
                                                  self.mean_loss_op,
                                                  self.mean_accuracy_op,
                                                  self.merged_summary],
                                                 feed_dict=feed_dict)
                loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                self.save_summary(summary, train_step + self.conf.reload_step, is_train=True)
            else:
                self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            # if train_step % self.conf.VAL_FREQ == 0:
            #     self.evaluate(train_step)

    def evaluate(self, train_step):
        print('start validating.......')
        self.sess.run(tf.local_variables_initializer())
        scan_input = np.zeros((0, self.conf.height, self.conf.width, self.conf.Dcut_size, self.conf.channel))
        scan_mask = np.zeros((0, self.conf.height, self.conf.width, self.conf.Dcut_size))
        scan_mask_pred = np.zeros((0, self.conf.height, self.conf.width, self.conf.Dcut_size))
        scan_num = 0
        hist = np.zeros((self.conf.num_cls, self.conf.num_cls))
        for step in range(self.num_val_batch):
            x_val, y_val = self.data_reader.next_batch(num=scan_num, mode='valid')
            for slice_num in range(x_val.shape[0]):     # for each slice of the validation image
                feed_dict = {self.inputs_pl: np.expand_dims(x_val[slice_num], 0),
                             self.labels_pl: np.expand_dims(y_val[slice_num], 0),
                             self.is_training_pl: False,
                             self.with_dropout_pl: False,
                             self.keep_prob_pl: 1}
                self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                input, mask, mask_pred = self.sess.run([self.inputs_pl,
                                                        self.labels_pl,
                                                        self.y_pred], feed_dict=feed_dict)
                scan_input = np.concatenate((scan_input, input), axis=0)
                scan_mask = np.concatenate((scan_mask, mask), axis=0)
                scan_mask_pred = np.concatenate((scan_mask_pred, mask_pred), axis=0)
                hist += get_hist(mask_pred.flatten(), mask.flatten(), num_cls=self.conf.num_cls)
            scan_num += 1
        IOU, ACC = compute_iou(hist)
        mean_IOU = np.mean(IOU)
        LABEL_NAMES = np.asarray(['background', 'liver', 'spleen', 'kidney', 'bone', 'vessel'])
        slice_idx = np.random.randint(low=0, high=scan_mask_pred.shape[0], size=10)
        depth_idx = np.random.randint(low=0, high=scan_mask_pred.shape[-1], size=10)
        dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        print('saving sample prediction images....... ')
        plot_save_preds(np.squeeze(scan_input[slice_idx, :, :, depth_idx, :]),
                        scan_mask[slice_idx, :, :, depth_idx],
                        scan_mask_pred[slice_idx, :, :, depth_idx],
                        dest_path, LABEL_NAMES)
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step, is_train=False)
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''
        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}{3}'
              .format(train_step, valid_loss, valid_acc, improved_str))
        print('- IOU: bg={0:.01%}, liver={1:.01%}, spleen={2:.01%}, '
              'kidney={3:.01%}, bone={4:.01%}, vessel={5:.01%}, Average={6:.01%}'
              .format(IOU[0], IOU[1], IOU[2], IOU[3], IOU[4], IOU[5], mean_IOU))
        print('- ACC: bg={0:.01%}, liver={1:.01%}, spleen={2:.01%}, '
              'kidney={3:.01%}, bone={4:.01%}, vessel={5:.01%}'
              .format(ACC[0], ACC[1], ACC[2], ACC[3], ACC[4], ACC[5]))
        print('-' * 60)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        print('loading the model.......')
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.numTest = self.data_reader.count_num_samples(mode='test')
        self.num_test_batch = int(self.numTest / self.conf.val_batch_size)
        print('start testing.......')
        self.sess.run(tf.local_variables_initializer())
        scan_input = np.zeros((0, self.conf.height, self.conf.width, self.conf.Dcut_size, self.conf.channel))
        scan_mask = np.zeros((0, self.conf.height, self.conf.width, self.conf.Dcut_size))
        scan_mask_pred = np.zeros((0, self.conf.height, self.conf.width, self.conf.Dcut_size))
        scan_num = 0
        hist = np.zeros((self.conf.num_cls, self.conf.num_cls))
        for step in range(self.num_test_batch):
            x_test, y_test = self.data_reader.next_batch(num=scan_num, mode='test')
            for slice_num in range(x_test.shape[0]):     # for each slice of the validation image
                feed_dict = {self.inputs_pl: np.expand_dims(x_test[slice_num], 0),
                             self.labels_pl: np.expand_dims(y_test[slice_num], 0),
                             self.keep_prob_pl: 1}
                self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                input, mask, mask_pred = self.sess.run([self.inputs_pl,
                                                        self.labels_pl,
                                                        self.y_pred], feed_dict=feed_dict)
                scan_input = np.concatenate((scan_input, input), axis=0)
                scan_mask = np.concatenate((scan_mask, mask), axis=0)
                scan_mask_pred = np.concatenate((scan_mask_pred, mask_pred), axis=0)
                hist += get_hist(mask_pred.flatten(), mask.flatten(), num_cls=self.conf.num_cls)
            scan_num += 1
        IOU, ACC = compute_iou(hist)
        mean_IOU = np.mean(IOU)
        LABEL_NAMES = np.asarray(['background', 'liver', 'spleen', 'kidney', 'bone', 'vessel'])
        slice_idx = np.random.randint(low=0, high=scan_mask_pred.shape[0], size=10)
        depth_idx = np.random.randint(low=0, high=scan_mask_pred.shape[-1], size=10)
        dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        print('saving sample prediction images....... ')
        plot_save_preds(np.squeeze(scan_input[slice_idx, :, :, depth_idx, :]),
                        scan_mask[slice_idx, :, :, depth_idx],
                        scan_mask_pred[slice_idx, :, :, depth_idx],
                        dest_path, LABEL_NAMES)
        print('-' * 25 + 'Validation' + '-' * 25)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.format(test_loss, test_acc))
        print('- IOU: bg={0:.01%}, liver={1:.01%}, spleen={2:.01%}, '
              'kidney={3:.01%}, bone={4:.01%}, vessel={5:.01%}, Average={6:.01%}'
              .format(IOU[0], IOU[1], IOU[2], IOU[3], IOU[4], IOU[5], mean_IOU))
        print('- ACC: bg={0:.01%}, liver={1:.01%}, spleen={2:.01%}, '
              'kidney={3:.01%}, bone={4:.01%}, vessel={5:.01%}'
              .format(ACC[0], ACC[1], ACC[2], ACC[3], ACC[4], ACC[5]))
        print('-' * 60)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')
