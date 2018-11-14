import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_3d, deconv_3d, max_pool
from utils import get_num_channels


class SegNet(BaseModel):
    def __init__(self, sess, conf):

        super(SegNet, self).__init__(sess, conf)
        self.k_size = self.conf.filter_size
        self.build_network(self.inputs_pl)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('SegNet'):
            with tf.variable_scope('Encoder'):
                # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
                x = conv_3d(x, self.k_size, 64, 'conv1_1', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 64, 'conv1_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = max_pool(x, ksize=2, stride=2, name='pool_1')

                # Second box of convolution layer(4)
                x = conv_3d(x, self.k_size, 128, 'conv2_1', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 128, 'conv2_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = max_pool(x, ksize=2, stride=2, name='pool_2')

                # Third box of convolution layer(7)
                x = conv_3d(x, self.k_size, 256, 'conv3_1', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 256, 'conv3_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 256, 'conv3_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = max_pool(x, ksize=2, stride=2, name='pool_3')

                # Fourth box of convolution layer(10)
                if self.bayes:
                    x = tf.layers.dropout(x, rate=(1 - self.keep_prob_pl), training=self.with_dropout_pl, name="dropout1")
                    x = conv_3d(x, self.k_size, 512, 'conv4_1', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                else:
                    x = conv_3d(x, self.k_size, 512, 'conv4_1', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 512, 'conv4_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 512, 'conv4_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = max_pool(x, ksize=2, stride=2, name='pool_4')

                # Fifth box of convolution layers(13)
                if self.bayes:
                    x = tf.layers.dropout(x, rate=(1-self.keep_prob_pl), training=self.with_dropout_pl, name="dropout2")
                    x = conv_3d(x, self.k_size, 512, 'conv5_1', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                else:
                    x = conv_3d(x, self.k_size, 512, 'conv5_1', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 512, 'conv5_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 512, 'conv5_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = max_pool(x, ksize=2, stride=2, name='pool_5')

            with tf.variable_scope('Decoder'):
                if self.bayes:
                    x = tf.layers.dropout(x, rate=(1-self.keep_prob_pl), training=self.with_dropout_pl, name="dropout3")
                    x = deconv_3d(x, 2, 512, 'deconv_5', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)
                else:
                    x = deconv_3d(x, 2, 512, 'deconv_5', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)
                x = conv_3d(x, self.k_size, 512, 'deconv5_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 512, 'deconv5_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 512, 'deconv5_4', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)

                # Second box of deconvolution layers(6)
                if self.bayes:
                    x = tf.layers.dropout(x, rate=(1-self.keep_prob_pl), training=self.with_dropout_pl, name="dropout4")
                    x = deconv_3d(x, 2, 512, 'deconv_4', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)

                else:
                    x = deconv_3d(x, 2, 512, 'deconv_4', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)
                x = conv_3d(x, self.k_size, 512, 'deconv4_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 512, 'deconv4_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 256, 'deconv4_4', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)

                # Third box of deconvolution layers(9)
                if self.bayes:
                    x = tf.layers.dropout(x, rate=(1-self.keep_prob_pl), training=self.with_dropout_pl, name="dropout5")
                    x = deconv_3d(x, 2, 256, 'deconv_3', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)
                else:
                    x = deconv_3d(x, 2, 256, 'deconv_3', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)
                x = conv_3d(x, self.k_size, 256, 'deconv3_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 256, 'deconv3_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 128, 'deconv3_4', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)

                # Fourth box of deconvolution layers(11)
                if self.bayes:
                    x = tf.layers.dropout(x, rate=(1-self.keep_prob_pl), training=self.with_dropout_pl, name="dropout6")
                    x = deconv_3d(x, 2, 128, 'deconv_2', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)
                else:
                    x = deconv_3d(x, 2, 128, 'deconv_2', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)

                x = conv_3d(x, self.k_size, 128, 'deconv2_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 64, 'deconv2_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)

                # Fifth box of deconvolution layers(13)
                x = deconv_3d(x, 2, 64, 'deconv_1', 2, add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl)
                x = conv_3d(x, self.k_size, 64, 'deconv1_2', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)
                x = conv_3d(x, self.k_size, 64, 'deconv1_3', self.conf.use_BN, self.is_training_pl, activation=tf.nn.relu)

            with tf.variable_scope('Classifier'):
                self.logits = conv_3d(x, 1, self.conf.num_cls, 'output', self.conf.use_BN, self.is_training_pl)
