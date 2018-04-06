import tensorflow as tf
import numpy as np


# Initialization function
def udi(shape):
    with tf.variable_scope('uniform_dist_initializer'):
        denominator = tf.cast((tf.reduce_prod(shape)), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)      
def ndi_2d(shape):
    with tf.variable_scope('normal_dist_initializer'):
        assert len(shape) == 4, 'Incorrect input shape'  # [filter_height, filter_width, in_channels, out_channels]
        return tf.truncated_normal(shape, mean=0,
                                   stddev=tf.sqrt(3. / (tf.reduce_prod(shape[:2]) * tf.reduce_sum(shape[2:]))))
def udi_2d(shape):
    with tf.variable_scope('uniform_dist_initializer'):
        assert len(shape) == 4, 'Incorrect input shape'  # [filter_height, filter_width, in_channels, out_channels]
        denominator = tf.cast((tf.reduce_prod(shape[:2]) * tf.reduce_sum(shape[2:])), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)
def ndi_3d(shape):
    with tf.variable_scope('normal_dist_initializer'):
        assert len(shape) == 5, 'Incorrect input shape'  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
                return tf.truncated_normal(shape, mean=0,
                                   stddev=tf.sqrt(3. / (tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:]))))
def udi_3d(shape):
    with tf.variable_scope('uniform_dist_initializer'):
        assert len(shape) == 5, 'Incorrect input shape'  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        denominator = tf.cast((tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:])), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)   


# convolution functions    
def convolution_2d(layer_input, filter, strides=[1,1,1,1], padding='VALID', dropout_rate=1, w_return_flag=False):
    assert len(filter) == 4  # [filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 4  # must match input dimensions [batch, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=udi_2d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    
    if w_return_flag is True:
        return tf.nn.conv2d(layer_input, w, strides, padding) + b, w
    else:
        return tf.nn.conv2d(layer_input, w, strides, padding) + b
    
def deconvolution_2d(layer_input, filter, output_shape, strides=[1,1,1,1], padding='VALID', dropout_rate=1, w_return_flag=False):
    assert len(filter) == 4  # [filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 4  # must match input dimensions [batch, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=udi_2d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')

    if w_return_flag is True:
        return tf.nn.conv2d_transpose(layer_input, w, output_shape, strides, padding) + b, w
    else:
        return tf.nn.conv2d_transpose(layer_input, w, output_shape, strides, padding) + b

def convolution_3d(layer_input, filter, strides=[1,1,1,1,1], padding='VALID', dropout_rate=1, w_return_flag=False):
    assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=udi_3d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    
    if w_return_flag is True:
        return tf.nn.conv3d(layer_input, w, strides, padding) + b, w
    else:
        return tf.nn.conv3d(layer_input, w, strides, padding) + b  
    
def deconvolution_3d(layer_input, filter, output_shape, strides=[1,1,1,1,1], padding='VALID', dropout_rate=1, w_return_flag=False):
    assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
    assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=udi_3d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')

    if w_return_flag is True:
        return tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding) + b, w
    else:
        return tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding) + b

# fully connected layer
def fc_layer(layer_input, in_channel, out_channel, afunc = tf.nn.relu):
    x = tf.reshape(layer_input,[-1,in_channel])
    w = tf.Variable(initial_value=init_uniform_dist([in_channel,out_channel]), name='weights')# shape unknown issue
    b = tf.Variable(tf.constant(1.0, shape=[out_channel]), name='biases')
    if afunc is None:
        return tf.matmul(x,w)+b
    else:
        return afunc(tf.matmul(x,w)+b)
    
# convolution layers
def convolution_block_2d(layer_input, n_channels, out_channel, f_shape = [3,3], strides = [1,1,1,1], num_convolutions = 1, afunc = tf.nn.relu, dropout_rate = 1.0):
    x = layer_input
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_2d(x, f_shape+[n_channels, n_channels], strides)
            if afunc is not None:
                x = afunc(x)
            if dropout_rate<1.0:
                x = tf.nn.dropout(x,keep_prob=1-dropout_rate)
                
    with tf.variable_scope('conv_' + str(num_convolutions - 1)):
        x = convolution_2d(x, f_shape+[n_channels, out_channel], strides)
        if afunc is not None:
                x = afunc(x)
        if dropout_rate<1.0:
            x = tf.nn.dropout(x,keep_prob=1-dropout_rate)
    return x

def convolution_block_3d(layer_input, n_channels, out_channel, f_shape = [3,3,3], strides = [1,1,1,1,1], num_convolutions = 1, afunc = tf.nn.relu, dropout_rate = 1.0):
    x = layer_input
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, f_shape+[n_channels, n_channels], strides)
            if afunc is not None:
                x = afunc(x)
            if dropout_rate<1.0:
                x = tf.nn.dropout(x,keep_prob=1-dropout_rate)
                
    with tf.variable_scope('conv_' + str(num_convolutions - 1)):
        x = convolution_2d(x, f_shape+[n_channels, out_channel], strides)
        if afunc is not None:
                x = afunc(x)
        if dropout_rate<1.0:
            x = tf.nn.dropout(x,keep_prob=1-dropout_rate)
    return x