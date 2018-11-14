import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_3d, deconv_3d, prelu
from model.utils.utils_segnet import max_pool, conv_layer
from utils import get_num_channels


class VNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=3,
                 num_convs=(1, 2, 3),
                 bottom_convs=3,
                 act_fcn=prelu):

        super(VNet, self).__init__(sess, conf)
        # super().__init__(sess, conf)  Python3
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.bottom_convs = bottom_convs
        self.k_size = self.conf.filter_size
        self.down_conv_factor = 2
        self.act_fcn = act_fcn
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('SegNet'):
            with tf.variable_scope('Encoder'):
                # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
                self.conv1_1 = conv_layer(x, "conv1_1", [3, 3, 3, 64], self.is_training_pl)
                self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.is_training_pl)
                self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

                # Second box of convolution layer(4)
                self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.is_training_pl)
                self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.is_training_pl)
                self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

                # Third box of convolution layer(7)
                self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.is_training_pl)
                self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.is_training_pl)
                self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.is_training_pl)
                self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

                # Fourth box of convolution layer(10)
                if self.bayes:
                    self.dropout1 = tf.layers.dropout(self.pool3, rate=(1 - self.keep_prob_pl),
                                                      training=self.with_dropout_pl, name="dropout1")
                    self.conv4_1 = conv_layer(self.dropout1, "conv4_1", [3, 3, 256, 512], self.is_training_pl,
                                              self.use_vgg,
                                              self.vgg_param_dict)
                else:
                    self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.is_training_pl,
                                              self.use_vgg,
                                              self.vgg_param_dict)
                self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
                self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
                self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

                # Fifth box of convolution layers(13)
                if self.bayes:
                    self.dropout2 = tf.layers.dropout(self.pool4, rate=(1 - self.keep_prob_pl),
                                                      training=self.with_dropout_pl, name="dropout2")
                    self.conv5_1 = conv_layer(self.dropout2, "conv5_1", [3, 3, 512, 512], self.is_training_pl,
                                              self.use_vgg,
                                              self.vgg_param_dict)
                else:
                    self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.is_training_pl,
                                              self.use_vgg,
                                              self.vgg_param_dict)
                self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
                self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
                self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

            with tf.variable_scope('Bottom_level'):
                x = self.conv_block_down(x, self.bottom_convs)

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        f = feature_list[l]
                        x = self.up_conv(x)
                        x = self.conv_block_up(x, f, self.num_convs[l])

            self.logits = conv_3d(x, 1, self.conf.num_cls, 'Output_layer', add_batch_norm=self.conf.use_BN,
                                  is_train=self.is_training)
