#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:01:32 2017

@author: pratik18v
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

FLAGS = None

import os
import glob
import cv2
import pickle
from progressbar import ProgressBar
import numpy as np

def wide_resnet(x, n=2, k=1):

    n_filters = {0:16, 1:16*k, 2:32*k, 3:64*k}
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    l_in = tf.reshape(x, [-1, 32, 32, 3])

    # output size: 32x32x16
    W_conv1 = weight_variable([3, 3, 3, 16])
    b_conv1 = bias_variable([16])
    l = tf.contrib.layers.batch_norm(conv2d(l_in, W_conv1) + b_conv1)

    keep_prob = tf.placeholder(tf.float32)
    
    # output size: 32x32x16
    l = residual_block(l, keep_prob, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, keep_prob, filters=n_filters[1])

    # output size: 16x16x32    
    l = residual_block(l, keep_prob, increase_dim=True, filters=n_filters[2])
    for _ in range(1,n+2):
        l = residual_block(l, keep_prob, filters=n_filters[2])

    # output size: 8x8x64
    l = residual_block(l, keep_prob, increase_dim=True, filters=n_filters[3])
    for _ in range(1,n+2):
        l = residual_block(l, keep_prob, filters=n_filters[3])

    bn_post_conv = tf.contrib.layers.batch_norm(l)
    bn_post_relu = tf.nn.relu(bn_post_conv)

    # average pooling
    avg_pool = tf.nn.avg_pool(bn_post_relu, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME')

    W_fc = weight_variable([8 * 8 * 64 * k, 3])
    b_fc = bias_variable([3])

    avg_pool_flat = tf.reshape(avg_pool, [-1, 8 * 8 * 64 * k])
    h_fc = tf.matmul(avg_pool_flat, W_fc) + b_fc

    return h_fc, keep_prob

def residual_block(l, keep_prob=0.5, increase_dim=False, projection=True, first=False, filters=16):
    input_num_filters = l.get_shape().as_list()[3]
    if increase_dim:
        first_stride = [1, 2, 2, 1]
    else:
        first_stride = [1, 1, 1, 1]

    if first:
        # hacky solution to keep layers correct
        bn_pre_relu = l
    else:
        # contains the BN -> ReLU portion, steps 1 to 2
        bn_pre_conv = tf.contrib.layers.batch_norm(l)
        bn_pre_relu = tf.nn.relu(bn_pre_conv)
    # contains the weight -> BN -> ReLU portion, steps 3 to 5
    W_conv1 = weight_variable([3, 3, input_num_filters, filters])
    b_conv1 = bias_variable([filters])
    conv_1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(bn_pre_relu, W_conv1, first_stride) + b_conv1))
    dropout = tf.nn.dropout(conv_1, keep_prob)
    # contains the last weight portion, step 6
    W_conv2 = weight_variable([3, 3, filters, filters])
    b_conv2 = bias_variable([filters])
    conv_2 = conv2d(dropout, W_conv2) + b_conv2
    # add shortcut connections
    if increase_dim:
        # projection shortcut, as option B in paper
        W_conv3 = weight_variable([1, 1, input_num_filters, filters])
        projection = conv2d(l, W_conv3, [1, 2, 2, 1])
        block = tf.add(conv_2, projection)
        
    elif first:
        W_conv3 = weight_variable([1, 1, input_num_filters, filters])
        projection = conv2d(l, W_conv3)
        block = tf.add(conv_2, projection)
        
    else:
        block = tf.add(conv_2, l)

    return block
    
def conv2d(x, W, stride = [1, 1, 1, 1]):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def generate_data(folder):

    pixel = 32
    if os.path.isfile('training-data.pkl') == False:
        imgnames = []
        for fname in glob.glob(folder+'*.jpg'):
            imgnames.append(fname)
        
        labels = []
        data = []
        pbar = ProgressBar()
        for i in pbar(range(len(imgnames))):
            img = cv2.imread(imgnames[i])
            #Bug with cv2.imread, not able to  read some images (8)
            if img == None:
                continue
            labels.append(int(imgnames[i][7])-1)
            #Resizing to pixel x pixel
            img_small = cv2.resize(img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
            #Flatten the image
            data.append(img_small.reshape(-1))
            
            #Data aumgentation
            #1. Flip image vertically
            labels.append(int(imgnames[i][7])-1)
            flip_img = cv2.flip(img,1)
            flip_img = cv2.resize(flip_img, (pixel,pixel), \
                                  interpolation = cv2.INTER_AREA)
            data.append(flip_img.reshape(-1))
            
            #2. Add gaussian noise
            labels.append(int(imgnames[i][7])-1)
            noise = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
            m = (50,50,50)
            s = (50,50,50)
            cv2.randn(noise,m,s)
            noise_img = img + noise
            noise_img = cv2.resize(noise_img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
            data.append(noise_img.reshape(-1))
            
            #3. Increasing contrast
            labels.append(int(imgnames[i][7])-1)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            contrast_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            contrast_img = cv2.resize(contrast_img, (pixel,pixel), \
                                      interpolation = cv2.INTER_AREA)
            data.append(contrast_img.reshape(-1))
            
            #4. Crop image
            labels.append(int(imgnames[i][7])-1)
            crop_img = img[int(0.1*img.shape[0]):img.shape[0]-int(0.1*img.shape[0]) \
                            ,int(0.1*img.shape[1]) :img.shape[1]-int(0.1*img.shape[0]), :]
            crop_img = cv2.resize(crop_img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
            data.append(crop_img.reshape(-1))
            
            #5. Rotate image by 20-degrees
            labels.append(int(imgnames[i][7])-1)
            M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),20,1)
            rot_img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
            rot_img = cv2.resize(rot_img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
            data.append(rot_img.reshape(-1))
                
        # Subtract mean and normalization
        mean = np.mean(data, axis=0)
        sd = np.std(data, axis = 0)
        data = np.asarray(data, dtype = np.float32)
        data -= mean
        data /= sd
        
        #Generating one-hot labels
        classes = 3
        labels = np.asarray(labels, dtype = np.int32)
        one_hot = np.zeros((labels.shape[0], classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
                
        with open('training-data.pkl', 'wb') as f:
            pickle.dump([data, labels, one_hot, mean, sd], f)
    
    else:
        with open('training-data.pkl', 'rb') as f:
            data, labels, one_hot, mean, sd = pickle.load(f)
    
    print('Number of images in class-1: {}'.format((labels == 0).sum()))
    print('Number of images in class-2: {}'.format((labels == 1).sum()))
    print('Number of images in class-3: {}'.format((labels == 2).sum()))
    
    #Make sure data dimensions are correct
    assert data.shape[0] == one_hot.shape[0]
    total = data.shape[0]
    
    #Shuffling data
    temp = zip(data,one_hot)
    np.random.shuffle(temp)
    data, one_hot = zip(*temp)    
    data = np.asarray(data, dtype = np.float32)
    one_hot = np.asarray(one_hot, dtype = np.int32)    
    
    #Splitting into train and test (3:1)
    trainX = data[:int(round(total*0.75)),:]
    trainY = one_hot[:int(round(total*0.75))]
    testX = data[int(round(total*0.75)):,:]
    testY = one_hot[int(round(total*0.75)):]
    
    #Some dimension checks
    assert trainX.shape[0] == trainY.shape[0]
    assert testX.shape[0] == testY.shape[0]
    
    print('Size of training set: {}'.format(trainX.shape[0]))
    print('Size of test set: {}'.format(testX.shape[0]))
    
    return trainX, trainY, testX, testY
    
data_index = 0
def generate_batch(X, y, batch_size):
    global data_index
    #print(data_index)
    if data_index >= X.shape[0]:
        data_index = 0
    batch = tuple([X[data_index:data_index+batch_size,:], y[data_index:data_index+batch_size]])
    data_index += batch_size
    return batch

def main(_):
    
    num_pixels = 32*32*3 #32*32*3
    
    #Loading data
    trainX, trainY, testX, testY = generate_data(FLAGS.data_dir)
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, num_pixels])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, FLAGS.classes])
    
    # Build the graph for the deep net
    y_conv, keep_prob = wide_resnet(x)
    
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.max_steps):
            batch = generate_batch(trainX, trainY, FLAGS.batch_size)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.dropout_prob})
            if i % 1000 == 0:
                print('test accuracy %g' % accuracy.eval(feed_dict={
                        x: testX, y_: testY, keep_prob: 1.0}))
    
        print('Final test accuracy %g' % accuracy.eval(feed_dict={
                        x: testX, y_: testY, keep_prob: 1.0}))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='images/',
                      help='Directory for storing input data')
    parser.add_argument('--classes', type=int, default=3,
                      help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=100,
                      help='Batch size')
    parser.add_argument('--max_steps', type=int, default=20500,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--dropout_prob', type=float, default=0.3,
                      help='Probability to drop units in dropout')
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
