import tensorflow as tf

from model.model_2D.base_model import BaseModel
from model.model_2D.ops import get_num_channels, conv_2d, BN_Relu_conv_2d, deconv_2d, max_pool


class Tiramisu(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=5,
                 num_convs=(4, 5, 7, 10, 12),
                 bottom_convs=15):

        super(Tiramisu, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.bottom_convs = bottom_convs
        self.k_size = self.conf.filter_size
        self.down_conv_factor = 2
        # BaseModel.__init__(self, sess, conf)

        # super().__init__(sess, conf)  Python3
        self.build_network(self.inputs_pl)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('Tiramisu'):
            feature_list = list()
            shape_list = list()

            with tf.variable_scope('input'):
                x = conv_2d(x, self.k_size, 48, 'input_layer', add_batch_norm=self.conf.use_BN,
                            is_train=self.is_training_pl, add_reg=self.conf.use_reg, activation=tf.nn.relu)
                # x = tf.nn.dropout(x, self.keep_prob)
                print('{}: {}'.format('input_layer', x.get_shape()))

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        level = self.dense_block(x, self.num_convs[l])
                        shape_list.append(tf.shape(level))
                        x = tf.concat((x, level), axis=-1)
                        print('{}: {}'.format('Encoder_level' + str(l + 1), x.get_shape()))
                        feature_list.append(x)
                        x = self.down_conv(x)

            with tf.variable_scope('Bottom_level'):
                x = self.dense_block(x, self.bottom_convs)
                print('{}: {}'.format('bottom_level', x.get_shape()))

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        x = self.up_conv(x)
                        stack = tf.concat((x, feature_list[l]), axis=-1)
                        print('{}: {}'.format('Decoder_level' + str(l + 1), x.get_shape()))
                        x = self.dense_block(stack, self.num_convs[l])
                        print('{}: {}'.format('Dense_block_level' + str(l + 1), x.get_shape()))
                        stack = tf.concat((stack, x), axis=-1)
                        print('{}: {}'.format('stck_depth' + str(l + 1), stack.get_shape()))

            with tf.variable_scope('output'):

                print('{}: {}'.format('out_block_input', stack.get_shape()))
                self.logits = BN_Relu_conv_2d(stack, 1, self.conf.num_cls, 'Output_layer',
                                              add_batch_norm=self.conf.use_BN,
                                              is_train=self.is_training_pl)
                print('{}: {}'.format('output', self.logits.get_shape()))

    def dense_block(self, layer_input, num_convolutions):
        x = layer_input
        layers = []
        for i in range(num_convolutions):
            layer = BN_Relu_conv_2d(inputs=x,
                                    filter_size=self.k_size,
                                    num_filters=self.conf.start_channel_num,
                                    layer_name='conv_' + str(i + 1),
                                    add_batch_norm=self.conf.use_BN,
                                    use_relu=True,
                                    is_train=self.is_training_pl)
            layer = tf.nn.dropout(layer, self.keep_prob_pl)
            layers.append(layer)
            x = tf.concat((x, layer), axis=-1)
        return tf.concat(layers, axis=-1)

    def down_conv(self, x):
        num_out_channels = get_num_channels(x)
        x = BN_Relu_conv_2d(inputs=x,
                            filter_size=1,
                            num_filters=num_out_channels,
                            layer_name='conv_down',
                            stride=1,
                            add_batch_norm=self.conf.use_BN,
                            is_train=self.is_training_pl,
                            use_relu=True)
        x = tf.nn.dropout(x, self.keep_prob_pl)
        x = max_pool(x, self.conf.pool_filter_size, stride=2, name='maxpool')
        return x

    def up_conv(self, x):
        num_out_channels = x.get_shape().as_list()[-1]
        x = deconv_2d(inputs=x,
                      filter_size=3,
                      num_filters=num_out_channels,
                      layer_name='conv_up',
                      stride=2,
                      add_batch_norm=False,
                      is_train=self.is_training_pl)
        return x