import glob

import tensorflow as tf
from tqdm import tqdm
from creat_TFRecord import createDataRecord, train_parser, valid_parser
from utils.plot_utils import plot_save_preds_2d
from utils.loss_utils import cross_entropy, dice_coeff, weighted_cross_entropy
from utils.eval_utils import get_hist, compute_iou, var_calculate_2d, get_uncertainty_measure
import os
import numpy as np


class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [None, self.conf.height, self.conf.width]
        self.create_placeholders()
        self.create_data_pipeline()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            # self.inputs_pl = tf.placeholder(tf.float32, self.input_shape, name='input')
            # self.labels_pl = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
            self.keep_prob_pl = tf.placeholder(tf.float32)
            self.handle = tf.placeholder(tf.string, shape=[])

    def create_data_pipeline(self):
        if self.conf.mode == 'train':
            if not os.path.exists(self.conf.train_tfrecords):
                print('Creating the train tfrecords file')
                input_address = glob.glob(self.conf.path_to_input_images_train)
                label_address = glob.glob(self.conf.path_to_output_labels_train)
                createDataRecord(self.conf.train_tfrecords, input_address, label_address)
            if not os.path.exists(self.conf.valid_tfrecords):
                print('Creating the validation tfrecords file')
                input_address = glob.glob(self.conf.path_to_input_images_val)
                label_address = glob.glob(self.conf.path_to_output_labels_val)
                createDataRecord(self.conf.valid_tfrecords, input_address, label_address)
            dataset_train = tf.data.TFRecordDataset(self.conf.train_tfrecords)
            dataset_valid = tf.data.TFRecordDataset(self.conf.valid_tfrecords)

            dataset_train = dataset_train.map(train_parser, num_parallel_calls=2 * self.conf.batch_size)
            dataset_valid = dataset_valid.map(valid_parser, num_parallel_calls=2 * self.conf.val_batch_size)

            dataset_train = dataset_train.repeat()
            dataset_train = dataset_train.shuffle(buffer_size=100)

            dataset_train = dataset_train.batch(self.conf.batch_size)
            dataset_valid = dataset_valid.batch(self.conf.val_batch_size)

            dataset_train = dataset_train.prefetch(1)  # make sure you always have one batch ready to serve
            dataset_valid = dataset_valid.prefetch(1)  # make sure you always have one batch ready to serve

            self.iterator = tf.data.Iterator.from_string_handle(self.handle, dataset_train.output_types)
            self.inputs_pl, self.labels_pl = self.iterator.get_next()
            self.inputs_pl.set_shape([None, self.conf.height, self.conf.width, self.conf.channel])
            self.labels_pl.set_shape([None, self.conf.height, self.conf.width])

            self.train_iterator = dataset_train.make_one_shot_iterator()
            self.valid_iterator = dataset_valid.make_initializable_iterator()

        elif self.conf.mode == 'test':
            if not os.path.exists(self.conf.test_tfrecords):
                print('Creating the test tfrecords file')
                input_address = glob.glob(self.conf.path_to_input_images_test)
                label_address = glob.glob(self.conf.path_to_output_labels_test)
                createDataRecord(self.conf.test_tfrecords, input_address, label_address)
            dataset_test = tf.data.TFRecordDataset(self.conf.test_tfrecords)
            dataset_test = dataset_test.map(valid_parser, num_parallel_calls=2 * self.conf.val_batch_size)
            dataset_test = dataset_test.batch(self.conf.val_batch_size)
            dataset_test = dataset_test.prefetch(1)  # make sure you always have one batch ready to serve
            self.iterator = tf.data.Iterator.from_string_handle(self.handle, dataset_test.output_types)
            self.inputs_pl, self.labels_pl = self.iterator.get_next()
            self.inputs_pl.set_shape([None, self.conf.height, self.conf.width, self.conf.channel])
            self.labels_pl.set_shape([None, self.conf.height, self.conf.width])
            self.test_iterator = dataset_test.make_initializable_iterator()

    def loss_func(self):
        with tf.name_scope('Loss'):
            self.y_prob = tf.nn.softmax(self.logits, axis=-1)
            y_one_hot = tf.one_hot(self.labels_pl, depth=self.conf.num_cls, axis=3, name='y_one_hot')
            if self.conf.weighted_loss:
                loss = weighted_cross_entropy(y_one_hot, self.logits, self.conf.num_cls, data=self.conf.data)
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
            self.y_pred = tf.argmax(self.logits, axis=3, name='decode_pred')
            correct_prediction = tf.equal(self.labels_pl, self.y_pred, name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=2000,
                                                   decay_rate=0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
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
                                         self.inputs_pl,
                                         max_outputs=5),
                        tf.summary.image('train/prediction_mask',
                                         tf.cast(tf.expand_dims(self.y_pred, -1),
                                                 tf.float32),
                                         max_outputs=5),
                        tf.summary.image('train/original_mask',
                                         tf.cast(tf.expand_dims(self.labels_pl, -1), tf.float32),
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
        self.best_validation_loss = 1000
        self.best_mean_IOU = 0
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
        else:
            print('----> Start Training')

        train_handle = self.sess.run(self.train_iterator.string_handle())
        for train_step in range(self.conf.reload_step, self.conf.reload_step + self.conf.max_step + 1):
            feed_dict = {self.is_training_pl: True,
                         self.keep_prob_pl: self.conf.keep_prob,
                         self.with_dropout_pl: True,
                         self.handle: train_handle}
            if train_step % self.conf.SUMMARY_FREQ == 0:
                _, _, _, summary, loss, acc = self.sess.run([self.train_op,
                                                             self.mean_loss_op,
                                                             self.mean_accuracy_op,
                                                             self.merged_summary,
                                                             self.mean_loss,
                                                             self.mean_accuracy],
                                                            feed_dict=feed_dict)
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                self.save_summary(summary, train_step, is_train=True)
            else:
                self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            if train_step % self.conf.VAL_FREQ == 0:
                print('-' * 25 + 'Validation' + '-' * 25)
                self.normal_evaluate(train_step=train_step)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        print('-' * 25 + 'Test' + '-' * 25)
        if not self.conf.bayes:
            self.normal_evaluate(train_step=step_num)
        else:
            self.MC_evaluate(dataset='test', train_step=step_num)
        # self.visualize(num_samples=20, train_step=step_num, mode='test')

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')

    def normal_evaluate(self, train_step=None):
        hist = np.zeros((self.conf.num_cls, self.conf.num_cls))
        plot_inputs = np.zeros((0, self.conf.height, self.conf.width, self.conf.channel))
        plot_mask = np.zeros((0, self.conf.height, self.conf.width))
        plot_mask_pred = np.zeros((0, self.conf.height, self.conf.width))

        if self.conf.mode == 'train':
            self.sess.run(self.valid_iterator.initializer)
            self.handle_ = self.sess.run(self.valid_iterator.string_handle())
        elif self.conf.mode == 'test':
            self.sess.run(self.test_iterator.initializer)
            self.handle_ = self.sess.run(self.test_iterator.string_handle())

        while True:
            try:
                feed_dict = {self.is_training_pl: True,
                             self.with_dropout_pl: False,
                             self.keep_prob_pl: 1,
                             self.handle: self.handle_}
                _, _, mask_pred, data_x, data_y, summary_valid = self.sess.run(
                    [self.mean_loss_op, self.mean_accuracy_op, self.y_pred,
                     self.inputs_pl, self.labels_pl, self.merged_summary],
                    feed_dict=feed_dict)
                hist += get_hist(mask_pred.flatten(), data_y.flatten(), num_cls=self.conf.num_cls)
                if plot_inputs.shape[0] < 100:  # randomly select a few slices to plot
                    # idx = np.random.randint(self.conf.batch_size)
                    plot_inputs = np.concatenate(
                        (plot_inputs, data_x.reshape(-1, self.conf.height, self.conf.width, self.conf.channel)), axis=0)
                    plot_mask = np.concatenate((plot_mask, data_y.reshape(-1, self.conf.height, self.conf.width)),
                                               axis=0)
                    plot_mask_pred = np.concatenate((plot_mask_pred,
                                                     mask_pred.reshape(-1, self.conf.height, self.conf.width)), axis=0)
            except tf.errors.OutOfRangeError:
                # assert len(validation_predictions) == VALIDATION_SIZE
                IOU, ACC = compute_iou(hist)
                mean_IOU = np.mean(IOU)
                loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                if self.conf.mode == "train":  # save the summaries and improved model in validation mode
                    # summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
                    self.save_summary(summary_valid, train_step, is_train=False)
                    if loss < self.best_validation_loss:
                        self.best_validation_loss = loss
                        print('>>>>>>>> model validation loss improved; saving the model......')
                        self.save(train_step)
                    elif mean_IOU > self.best_mean_IOU:
                        self.best_mean_IOU = mean_IOU
                        print('>>>>>>>> model mean IOU improved; saving the model......')
                        self.save(train_step)

                print('****** IoU & ACC ******')
                print('Mean IoU = {0:.01%}'.format(mean_IOU))
                for ii in range(self.conf.num_cls):
                    print('     - {0:<15}: IoU={1:<5.01%}, ACC={2:<5.01%}'.format(self.conf.label_name[ii], IOU[ii],
                                                                                  ACC[ii]))
                print('-' * 20)
                self.visualize(plot_inputs, plot_mask, plot_mask_pred, train_step=train_step)
                break

    def MC_evaluate(self, dataset='valid', train_step=None):
        num_batch = self.num_test_batch if dataset == 'test' else self.num_val_batch
        hist = np.zeros((self.conf.num_cls, self.conf.num_cls))
        self.sess.run(tf.local_variables_initializer())
        all_inputs = np.zeros((0, self.conf.height, self.conf.width, self.conf.channel))
        all_mask = np.zeros((0, self.conf.height, self.conf.width))
        all_pred = np.zeros((0, self.conf.height, self.conf.width))
        all_var = np.zeros((0, self.conf.height, self.conf.width))
        cls_uncertainty = np.zeros((0, self.conf.height, self.conf.width, self.conf.num_cls))
        for step in tqdm(range(num_batch)):
            start = self.conf.val_batch_size * step
            end = self.conf.val_batch_size * (step + 1)
            data_x, data_y = self.data_reader.next_batch(start=start, end=end, mode=dataset)
            mask_pred_mc = [np.zeros((self.conf.val_batch_size, self.conf.height, self.conf.width))
                            for _ in range(self.conf.monte_carlo_simulations)]
            mask_prob_mc = [np.zeros((self.conf.val_batch_size, self.conf.height, self.conf.width, self.conf.num_cls))
                            for _ in range(self.conf.monte_carlo_simulations)]
            feed_dict = {self.inputs_pl: data_x,
                         self.labels_pl: data_y,
                         self.is_training_pl: True,
                         self.with_dropout_pl: True,
                         self.keep_prob_pl: self.conf.keep_prob}
            for mc_iter in range(self.conf.monte_carlo_simulations):
                inputs, mask, mask_prob, mask_pred = self.sess.run([self.inputs_pl,
                                                                    self.labels_pl,
                                                                    self.y_prob,
                                                                    self.y_pred], feed_dict=feed_dict)
                mask_prob_mc[mc_iter] = mask_prob
                mask_pred_mc[mc_iter] = mask_pred

            prob_mean = np.nanmean(mask_prob_mc, axis=0)
            prob_variance = np.var(mask_prob_mc, axis=0)
            pred = np.argmax(prob_mean, axis=-1)
            var_one = np.nanmean(prob_variance, axis=-1)
            # var_one = var_calculate_2d(pred, prob_variance)
            hist += get_hist(pred.flatten(), mask.flatten(), num_cls=self.conf.num_cls)

            # if all_inputs.shape[0] < 6:
            # ii = np.random.randint(self.conf.val_batch_size)
            # ii = 1
            all_inputs = np.concatenate((all_inputs, inputs.reshape(-1, self.conf.height, self.conf.width,
                                                                    self.conf.channel)), axis=0)
            all_mask = np.concatenate((all_mask, mask.reshape(-1, self.conf.height, self.conf.width)), axis=0)
            all_pred = np.concatenate((all_pred, pred.reshape(-1, self.conf.height, self.conf.width)), axis=0)
            all_var = np.concatenate((all_var, var_one.reshape(-1, self.conf.height, self.conf.width)), axis=0)
            cls_uncertainty = np.concatenate((cls_uncertainty,
                                              prob_variance.reshape(-1, self.conf.height, self.conf.width,
                                                                    self.conf.num_cls)),
                                             axis=0)
            # else:
        self.visualize(all_inputs, all_mask, all_pred, all_var, cls_uncertainty, train_step=train_step)
        import h5py
        h5f = h5py.File(self.conf.run_name + '_bayes.h5', 'w')
        h5f.create_dataset('x', data=all_inputs)
        h5f.create_dataset('y', data=all_mask)
        h5f.create_dataset('y_pred', data=all_pred)
        h5f.create_dataset('y_var', data=all_var)
        h5f.create_dataset('cls_uncertainty', data=cls_uncertainty)
        h5f.close()

        uncertainty_measure = get_uncertainty_measure(all_inputs, all_mask, all_pred, all_var)

        # break
        IOU, ACC = compute_iou(hist)
        mean_IOU = np.mean(IOU)
        print('****** IoU & ACC ******')
        print('Uncertainty Quality Measure = {}'.format(uncertainty_measure))
        print('Mean IoU = {0:.01%}'.format(mean_IOU))
        for ii in range(self.conf.num_cls):
            print('     - {0} class: IoU={1:.01%}, ACC={2:.01%}'.format(self.conf.label_name[ii], IOU[ii], ACC[ii]))
        print('-' * 20)

    def visualize(self, x, y, y_pred, var=None, cls_uncertainty=None, train_step=None):
        # all of shape (#images, 512, 512)
        if self.conf.mode == 'train':
            dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step))
        elif self.conf.mode == "test":
            dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step) + '_test')

        print('saving sample prediction images....... ')
        cls_uncertainty = None  #################
        if not self.conf.bayes or self.conf.mode == 'train':
            # run it either in validation mode or when non-bayesian network
            plot_save_preds_2d(x, y, y_pred, path=dest_path, label_names=np.array(self.conf.label_name))
        else:
            if cls_uncertainty is None:
                plot_save_preds_2d(x, y, y_pred, var, path=dest_path,
                                   label_names=np.array(self.conf.label_name))
            else:
                plot_save_preds_2d(x, y, y_pred, var, cls_uncertainty, path=dest_path,
                                   label_names=np.array(self.conf.label_name))

        print('Images saved in {}'.format(dest_path))
        print('-' * 20)
