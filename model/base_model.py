import tensorflow as tf
from Data_Loader import DataLoader
from utils import cross_entropy, dice_coeff, compute_iou, weighted_cross_entropy
import os
import numpy as np


class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.is_training = True
        self.input_shape = [None, None, None, None, self.conf.channel]
        self.output_shape = [None, None, None, None]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.keep_prob = tf.placeholder(tf.float32)
            # self.is_training = tf.placeholder_with_default(True, shape=(), name="is_train")

    def loss_func(self):
        with tf.name_scope('Loss'):
            y_one_hot = tf.one_hot(self.y, depth=self.conf.num_cls, axis=4, name='y_one_hot')
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
            correct_prediction = tf.equal(self.y, self.y_pred, name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=500,
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
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss),
                        tf.summary.scalar('accuracy', self.mean_accuracy),
                        tf.summary.image('train/original_image',
                                         self.x[:, :, :, int(self.conf.depth/2)],
                                         max_outputs=self.conf.batch_size),
                        tf.summary.image('train/prediction_mask',
                                         tf.cast(tf.expand_dims(self.y_pred[:, :, :, int(self.conf.depth/2)], -1),
                                                 tf.float32),
                                         max_outputs=self.conf.batch_size),
                        tf.summary.image('train/original_mask',
                                         tf.cast(tf.expand_dims(self.y[:, :, :, int(self.conf.depth/2)], -1), tf.float32),
                                         max_outputs=self.conf.batch_size)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step):
        # print('----> Summarizing at step {}'.format(step))
        if self.is_training:
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
        # self.numValid = self.data_reader.count_num_samples(mode='valid')
        # self.num_val_batch = int(self.numValid / self.conf.val_batch_size)
        for train_step in range(1, self.conf.max_step + 1):
            # print('Step: {}'.format(train_step))
            self.is_training = True
            if train_step % self.conf.SUMMARY_FREQ == 0:
                x_batch, y_batch = self.data_reader.next_batch(mode='train')
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: 0.5}
                _, _, _, summary = self.sess.run([self.train_op,
                                                  self.mean_loss_op,
                                                  self.mean_accuracy_op,
                                                  self.merged_summary],
                                                 feed_dict=feed_dict)
                loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                self.save_summary(summary, train_step + self.conf.reload_step)
            else:
                x_batch, y_batch = self.data_reader.next_batch(mode='train')
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: 0.5}
                self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            # if train_step % self.conf.VAL_FREQ == 0:
            #     self.is_training = False
            #     self.evaluate(train_step)

    def evaluate(self, train_step):
        self.sess.run(tf.local_variables_initializer())
        all_y = np.zeros((0, self.conf.height, self.conf.width, self.conf.depth))
        all_y_pred = np.zeros((0, self.conf.height, self.conf.width, self.conf.depth))
        for step in range(self.num_val_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.keep_prob: 1}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            y, y_pred = self.sess.run([self.y, self.y_pred], feed_dict=feed_dict)
            all_y = np.concatenate((all_y, y), axis=0)
            all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)
        IOU = compute_iou(all_y_pred, all_y, num_cls=self.conf.num_cls)
        mean_IOU = np.mean(IOU)
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step)
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''
        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}{3}'
              .format(train_step, valid_loss, valid_acc, improved_str))
        print('BackGround={0:.01%}, Neuron={1:.01%}, Vessel={2:.01%}, Average={3:.01%}'
              .format(IOU[0], IOU[1], IOU[2], mean_IOU))
        print('-' * 60)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.numTest = self.data_reader.count_num_samples(mode='test')
        self.num_test_batch = int(self.numTest / self.conf.val_batch_size)
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        all_y = np.zeros((0, self.conf.height, self.conf.width, self.conf.depth))
        all_y_pred = np.zeros((0, self.conf.height, self.conf.width, self.conf.depth))
        for step in range(self.num_test_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.keep_prob: 1}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            y, y_pred = self.sess.run([self.y, self.y_pred], feed_dict=feed_dict)
            all_y = np.concatenate((all_y, y), axis=0)
            all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)
        IOU = compute_iou(all_y_pred, all_y, num_cls=self.conf.num_cls)
        mean_IOU = np.mean(IOU)
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.format(test_loss, test_acc))
        print('BackGround={0:.01%}, Neuron={1:.01%}, Vessel={2:.01%}, Average={3:.01%}'
              .format(IOU[0], IOU[1], IOU[2], mean_IOU))
        print('-' * 50)

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