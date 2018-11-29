import tensorflow as tf
from model.model_2D.base_model import BaseModel
from model.model_2D.ops import conv_2d, deconv_2d, BN_Relu_conv_2d, max_pool, batch_norm, Relu, avg_pool, concatenation


class DenseNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=5,
                 num_blocks=(4, 6, 8, 10, 12),  # number of bottleneck blocks at each level
                 bottom_convs=16):  # number of convolutions at the bottom of the network
        assert num_levels == len(num_blocks), "number of levels doesn't match with number of blocks!"
        super(DenseNet, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.num_blocks = num_blocks
        self.bottom_convs = bottom_convs
        self.k = self.conf.growth_rate
        self.theta_down = self.conf.theta_down
        self.theta_up = self.conf.theta_up
        self.trans_out = 4 * self.k
        self.down_conv_factor = 2
        self.build_network(self.inputs_pl)
        self.configure_network()

    def build_network(self, x_input):
        # Building network...
        with tf.variable_scope('DenseNet'):
            feature_list = list()
            shape_list = list()
            x = conv_2d(x_input, filter_size=3, num_filters=self.trans_out, stride=1, layer_name='conv1',
                        add_batch_norm=self.conf.use_BN, is_train=self.is_training_pl, add_reg=self.conf.use_reg)
            print('conv1 shape: {}'.format(x.get_shape()))
            shape_list.append(tf.shape(x))

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        x = self.dense_block(x, self.num_blocks[l], scope='DB_' + str(l + 1))
                        feature_list.append(x)
                        print('DB_{} shape: {}'.format(str(l + 1), x.get_shape()))
                        x = self.transition_down(x, scope='TD_' + str(l + 1))
                        print('TD_{} shape: {}'.format(str(l + 1), x.get_shape()))
                        if l != self.num_levels - 1:
                            shape_list.append(tf.shape(x))

            with tf.variable_scope('Bottom_level'):
                x = self.dense_block(x, self.bottom_convs, scope='BottomBlock')
                print('bottom_level shape: {}'.format(x.get_shape()))

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        shape = x.get_shape().as_list()
                        # out_shape = [self.conf.batch_size] + list(map(lambda x: x*2, shape[1:-1])) \
                        #             + [int(shape[-1]*self.theta_up)]
                        # out_shape = tf.shape(tf.zeros(out_shape))
                        x = self.transition_up(x, scope='TU_' + str(l + 1), num_filters=int(shape[-1]*self.theta_up))
                        print('TU_{} shape: {}'.format(str(l + 1), x.get_shape()))
                        stack = tf.concat((x, feature_list[l]), axis=-1)
                        print('After concat shape: {}'.format(stack.get_shape()))
                        x = self.dense_block(stack, self.num_blocks[l], scope='DB_' + str(l + 1))
                        print('DB_{} shape: {}'.format(str(l + 1), x.get_shape()))

            with tf.variable_scope('output'):
                x = BN_Relu_conv_2d(x, 3, 256, 'pre_output_layer', add_reg=self.conf.use_reg,
                                    is_train=self.is_training_pl)
                print('pre_out shape: {}'.format(x.get_shape()))
                self.logits = BN_Relu_conv_2d(x, 1, self.conf.num_cls, 'Output_layer',
                                              add_reg=self.conf.use_reg, is_train=self.is_training_pl)
                print('{}: {}'.format('output', self.logits.get_shape()))

    def dense_block(self, layer_input, num_blocks, scope):
        with tf.name_scope(scope):
            layers_concat = list()
            layers_concat.append(layer_input)
            x = self.bottleneck_block(layer_input, scope=scope + '_BB_' + str(0))
            layers_concat.append(x)
            for i in range(num_blocks - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_block(x, scope=scope + '_BB_' + str(i + 1))
                layers_concat.append(x)
            x = concatenation(layers_concat)
        return x

    def bottleneck_block(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_norm(x, is_training=self.is_training_pl, scope='BN1')
            x = Relu(x)
            x = conv_2d(x, filter_size=1, num_filters=4 * self.k, add_batch_norm=self.conf.use_BN,
                        layer_name='conv1', add_reg=self.conf.use_reg, is_train=self.is_training_pl)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob_pl)

            x = batch_norm(x, is_training=self.is_training_pl, scope='BN2')
            x = Relu(x)
            x = conv_2d(x, filter_size=3, num_filters=self.k, add_batch_norm=self.conf.use_BN,
                        layer_name='conv2', add_reg=self.conf.use_reg, is_train=self.is_training_pl)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob_pl)
            return x

    def transition_down(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_norm(x, is_training=self.is_training_pl, scope='BN')
            x = Relu(x)
            x = conv_2d(x, filter_size=1, num_filters=int(x.get_shape().as_list()[-1]*self.theta_down),
                        layer_name='conv', add_reg=self.conf.use_reg, add_batch_norm=self.conf.use_BN,
                        is_train=self.is_training_pl)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob_pl)
            x = avg_pool(x, ksize=2, stride=2, scope='avg_pool')
            return x

    def transition_up(self, x, scope, num_filters=None):
        with tf.variable_scope(scope):
            x = batch_norm(x, is_training=self.is_training_pl, scope='BN_1')
            x = Relu(x)
            x = conv_2d(x, filter_size=1, num_filters=int(x.get_shape().as_list()[-1]*self.theta_up),
                        layer_name='conv', add_reg=self.conf.use_reg, add_batch_norm=self.conf.use_BN,
                        is_train=self.is_training_pl)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob_pl)
            x = batch_norm(x, is_training=self.is_training_pl, scope='BN_2')
            x = Relu(x)
            if not num_filters:
                num_filters = self.trans_out
            x = deconv_2d(inputs=x,
                          filter_size=3,
                          num_filters=num_filters,
                          layer_name='deconv',
                          stride=2,
                          add_reg=self.conf.use_reg,
                          add_batch_norm=False,
                          is_train=self.is_training_pl)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob_pl)
        return x