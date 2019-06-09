import tensorflow as tf
import numpy as np
from DL_app.tf_util import *


def v_net(tf_input, input_channels, output_channels=1, n_channels=16):

    with tf.variable_scope('contracting_path'):

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        if input_channels == 1:
            tf_input = tf.tile(tf_input, [1, 1, 1, 1, 1])
            
        with tf.variable_scope('level_0'):
            c0 = convolution_3d(tf_input, [5, 5, 5, input_channels, n_channels], [1, 1, 1, 1, 1])

        with tf.variable_scope('level_1'):
            c1 = convolution_block(c0, n_channels, n_channels,  num_convolutions = 1)
            c12 = down_convolution_3d(c1, n_channels)

        with tf.variable_scope('level_2'):
            c2 = convolution_block(c12, n_channels * 2, n_channels * 2, num_convolutions = 2)
            c22 = down_convolution_3d(c2, n_channels * 2)

        with tf.variable_scope('level_3'):
            c3 = convolution_block(c22, n_channels * 4, num_convolutions = 3)
            c32 = down_convolution_3d(c3, n_channels * 4)

        with tf.variable_scope('level_4'):
            c4 = convolution_block(c32, n_channels * 8, num_convolutions = 3)
            c42 = down_convolution_3d(c4, n_channels * 8)

        with tf.variable_scope('level_5'):
            c5 = convolution_block(c42, n_channels * 16, num_convolutions = 3)
            c52 = up_convolution_3d(c5, c4.shape, n_channels * 16)

    with tf.variable_scope('expanding_path'):

        with tf.variable_scope('level_4'):
            e4 = concat_layer(c52, c4, n_channels * 8)
            e4 = convolution_block(e4, n_channels * 8, num_convolutions = 3)
            e42 = up_convolution_3d(e4, c3.shape, n_channels * 8)

        with tf.variable_scope('level_3'):
            e3 = concat_layer(e42, c3, n_channels * 4)
            e3 = convolution_block(e3, n_channels * 4, num_convolutions = 3)
            e32 = up_convolution(e3, c2.shape, n_channels * 4)

        with tf.variable_scope('level_2'):
            e2 = concat_layer(e32, c2, n_channels * 2)
            e2 = convolution_block(e2, n_channels * 2, num_convolutions = 2)
            e22 = up_convolution(e2, tf.shape(c1), n_channels * 2)

        with tf.variable_scope('level_1'):
            e1 = concat_layer(e22, c1, n_channels, 1)
            e1 = convolution_block(e1, n_channels, num_convolutions = 1)
        
        with tf.variable_scope('output_layer'):
            output = convolution_3d(e1, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])

    return output