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
        return tf.truncated_normal(shape, mean=0, stddev=tf.sqrt(3. / (tf.reduce_prod(shape[:2]) * tf.reduce_sum(shape[2:]))))
    
def udi_2d(shape):
    with tf.variable_scope('uniform_dist_initializer'):
        assert len(shape) == 4, 'Incorrect input shape'  # [filter_height, filter_width, in_channels, out_channels]
        denominator = tf.cast((tf.reduce_prod(shape[:2]) * tf.reduce_sum(shape[2:])), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)
    
def ndi_3d(shape):
    with tf.variable_scope('normal_dist_initializer'):
        assert len(shape) == 5, 'Incorrect input shape'  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        return tf.truncated_normal(shape, mean=0,stddev=tf.sqrt(3. / (tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:]))))
        

def udi_3d(shape):
    with tf.variable_scope('uniform_dist_initializer'):
        assert len(shape) == 5, 'Incorrect input shape'  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        denominator = tf.cast((tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:])), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)   


# convolution functions    
def convolution_2d(layer_input, filter, strides=[1,1,1,1], padding='SAME', dropout_rate=1, w_return_flag=False):
    assert len(filter) == 4  # [filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 4  # must match input dimensions [batch, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=udi_2d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    
    if w_return_flag is True:
        return tf.nn.conv2d(layer_input, w, strides, padding) + b, w
    else:
        return tf.nn.conv2d(layer_input, w, strides, padding) + b
    
def deconvolution_2d(layer_input, filter, output_shape, strides=[1,1,1,1], padding='SAME', dropout_rate=1, w_return_flag=False):
    assert len(filter) == 4  # [filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 4  # must match input dimensions [batch, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=udi_2d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')

    if w_return_flag is True:
        return tf.nn.conv2d_transpose(layer_input, w, output_shape, strides, padding) + b, w
    else:
        return tf.nn.conv2d_transpose(layer_input, w, output_shape, strides, padding) + b

def convolution_3d(layer_input, filter, strides=[1,1,1,1,1], padding='SAME', dropout_rate=1, w_return_flag=False):
    assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=udi_3d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    
    if w_return_flag is True:
        return tf.nn.conv3d(layer_input, w, strides, padding) + b, w
    else:
        return tf.nn.conv3d(layer_input, w, strides, padding) + b  
    
def deconvolution_3d(layer_input, filter, output_shape, strides=[1,1,1,1,1], padding='SAME', dropout_rate=1, w_return_flag=False):
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
    x = tf.matmul(x,w)+b
    if afunc is not None:
        x = afunc(x)
        
    return x
    
# convolution layers
def convolution_block_2d(layer_input, n_channels, out_channel, f_shape = [3,3], strides = [1,1,1,1], num_convolutions = 1, afunc = tf.nn.relu, dropout_rate = 1.0):
    x = layer_input
    i_channels = x.shape.as_list()[-1]
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_2d(x, f_shape+[i_channels, n_channels], strides)
            i_channels = n_channels
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
    i_channels = x.shape.as_list()[-1]
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, f_shape+[i_channels, n_channels], strides)
            i_channels = n_channels
            if afunc is not None:
                x = afunc(x)
            if dropout_rate<1.0:
                x = tf.nn.dropout(x,keep_prob=1-dropout_rate)
                
    with tf.variable_scope('conv_' + str(num_convolutions - 1)):
        x = convolution_3d(x, f_shape+[n_channels, out_channel], strides)
        if afunc is not None:
                x = afunc(x)
        if dropout_rate<1.0:
            x = tf.nn.dropout(x,keep_prob=1-dropout_rate)
    return x

def concat_layer_3d(input1, input2, n_channels, num_convolutions = 1, afunc = None):
    # activation function issue
    x = tf.concat((input1, input2), axis=-1)
    i_channels = x.shape.as_list()[-1]
    with tf.variable_scope('catconv_' + str(1)):
        x = convolution_3d(x, [3, 3, 3, i_channels, n_channels], [1, 1, 1, 1, 1])

    for i in range(num_convolutions - 1):
        with tf.variable_scope('catconv_' + str(i+1)):
            x = convolution_3d(x, [3, 3, 3, n_channels, n_channels], [1, 1, 1, 1, 1])
            
    if afunc is not None:        
        x = afunc(x)
    return x

def concat_layer_2d(input1, input2, n_channels, num_convolutions = 1, afunc = None):
    # activation function issue
    x = tf.concat((input1, input2), axis=-1)
    i_channels = x.shape.as_list()[-1]
    with tf.variable_scope('catconv_' + str(1)):
        x = convolution_2d(x, [3, 3, i_channels, n_channels], [1, 1, 1, 1])

    for i in range(num_convolutions - 1):
        with tf.variable_scope('catconv_' + str(i+1)):
            x = convolution_2d(x, [3, 3, n_channels, n_channels], [1, 1, 1, 1])
            
    if afunc is not None:        
        x = afunc(x)
    return x

def down_convolution_3d(layer_input, in_channels,afunc = tf.nn.relu):
    with tf.variable_scope('down_convolution'):
        x = convolution_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1])
        
    if afunc is not None:        
        x = afunc(x)
        
    return x
    
def up_convolution_3d(layer_input, output_shape, in_channels,afunc = tf.nn.relu):
    with tf.variable_scope('up_convolution'):
        x = deconvolution_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels], output_shape, [1, 2, 2, 2, 1])
        
    if afunc is not None:        
        x = afunc(x)
        
    return x
    
