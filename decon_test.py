import tensorflow as tf
import numpy as np
from DL_app.tf_util import *
import time

def decode_net_2d(Ax_placeholder, ATAx_placeholder):
    # Input : Ax, ATAx
    # Output: x
    Ax = Ax_placeholder
    ATAx = ATAx_placeholder
    with tf.variable_scope('Ax_net'):
        out_channel = 4
        n_channel = 4
        Ax = convolution_block_2d(Ax, n_channel, out_channel, f_shape = [5,5], strides = [1,1,1,1], num_convolutions = 3)
    with tf.variable_scope('ATAx_net'):
        out_channel = 4
        n_channel = 8
        ATAx = convolution_block_2d(ATAx, n_channel, out_channel, f_shape = [5,5], strides = [1,1,1,1], num_convolutions = 3)
    with tf.variable_scope('Comb_net'):
        out_channel = 2
        n_channel = 16
        C2 = concat_layer_2d(Ax, ATAx, n_channel)
        cout = convolution_block_2d(C2, n_channel, out_channel, f_shape = [5,5], strides = [1,1,1,1], num_convolutions = 3)

    return cout


def decode_net(filter_placeholder, image_placeholder):

    # filter:[9,9]
    with tf.variable_scope('filter_net'):
        out_channel = 8
        n_channel = 4
        f1 = convolution_block_2d(filter_placeholder, n_channel, out_channel, f_shape = [5,5], strides = [1,1,1,1], num_convolutions = 3)
        f2 = tf.reshape(f1,[9,9,out_channel,1])

    with tf.variable_scope('image_net'):
        out_channel = 8
        n_channel = 4
        input1 = convolution_block_2d(image_placeholder, n_channel, out_channel, f_shape = [5,5], strides = [1,1,1,1], num_convolutions = 3)

    with tf.variable_scope('comb_net'):
        out_channel = 2
        n_channel = 16
        c2 = tf.nn.conv2d(input1, f2, strides = [1,1,1,1], padding = 'VALID')
        cout = convolution_block_2d(c2, n_channel, out_channel, f_shape = [5,5], strides = [1,1,1,1], num_convolutions = 3)

    return cout


Img_shape = [512,512]

Img0 = tf.placeholder(tf.float32, shape = [None, Img_shape[0], Img_shape[1], 2])
AI = tf.placeholder(tf.float32, shape = [None, Img_shape[0], Img_shape[1], 2])
dAI = tf.placeholder(tf.float32, shape = [None, Img_shape[0], Img_shape[1], 4])


pred_res = decode_net_2d(AI, dAI)
loss = tf.nn.l2_loss(pred_res-(AI-Img0))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = 1e-4

with tf.control_dependencies(update_ops):
    #global_step = tf.train.get_or_create_global_step()
    global_step = tf.contrib.framework.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    ops = optimizer.minimize(loss, global_step=global_step)

tf.summary.scalar('Loss', loss)


import DL_app.Img_synthesis1 as Is
import IO.read_dcm as Dcm
from os import listdir
import random
# load data

t = time.ctime()
output_dir = './result_' + t
nepoch = 1

file_dir = '../../data/E6317/3/'
files = listdir(file_dir)
# random.shuffle(files)
nepoch = 50
batch_size = 20
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .4
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    print('\n\nTraining Statistics:\n')
    sess.run(tf.global_variables_initializer())

    train_summary_writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(output_dir + '/test', sess.graph)

    summary_loss = []

    for epoch in range(nepoch):
        # data prep
        I0, Id, gx, gy = [],[],[],[]
        # wrap up as a function
        for dcm_file in files[40:80]:
            Imgt = Dcm.read_dcm(file_dir+dcm_file)
            I0t, Idt =Is.MRI_undersample_2d(np.squeeze(Imgt),full_area = [32,32], N = 40, sigma = 1e-5)
            gxt = np.gradient(Idt,axis = 1)
            gyt = np.gradient(Idt,axis = 2)
            I0.append([I0t])
            Id.append([Idt])
            gx.append([gxt])
            gy.append([gyt])

        # [N, nx, ny]
        I0 = np.vstack(I0).reshape([-1]+Img_shape)
        Id = np.vstack(Id).reshape([-1]+Img_shape)
        gx = np.vstack(gx).reshape([-1]+Img_shape)
        gy = np.vstack(gy).reshape([-1]+Img_shape)
        index = np.arange(I0.shape[0])
        random.shuffle(index)
        print('Data Prep Done!')
        # batch 
        for i in range((index.size-1)//batch_size):
            batch_ind = index[i*batch_size:np.minimum((i+1)*batch_size,index.size)]

            Img0_b = np.stack([np.real(I0[batch_ind,:]),np.imag(I0[batch_ind,:])],axis=3)
            AI_b = np.stack([np.real(Id[batch_ind,:]),np.imag(Id[batch_ind,:])],axis=3)
            dAI_b = np.stack([np.real(gx[batch_ind,:]),np.imag(gx[batch_ind,:]),np.real(gy[batch_ind,:]),np.imag(gy[batch_ind,:])],axis=3)

            _, summary_loss_t, step  = sess.run([ops, loss, global_step],feed_dict={Img0: Img0_b,AI: AI_b, dAI: dAI_b})
            if i % 10 is 0:
                print('loss is :',summary_loss_t)
                summary_loss.append([summary_loss_t])
		saver.save(sess, output_dir+'/model.ckpy')
