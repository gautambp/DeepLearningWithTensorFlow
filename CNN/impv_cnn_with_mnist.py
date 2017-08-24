# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read mnist (handwriting) images
# each image is 28x28 = 784 pixels
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

img_width = 28
img_height = 28
pixel_count = img_width * img_height
no_output = 10

# CNN Architecture
# 784 inputs (for each image pixel)
# 10 outputs (1 for each digit)

session = tf.InteractiveSession()

# create placeholders for inputs and outputs
x = tf.placeholder(tf.float32, shape=[None, pixel_count])
y_ = tf.placeholder(tf.float32, shape=[None, no_output])

# convert to image data structure
# -1 : batch no
# width, height, 1 (for grayscale img - as only one channel)
x_image = tf.reshape(x, [-1,img_width,img_height,1])

# weight for conv1 layer - (32 filters - size 5x5)
# shape - [filter_height, filter_width, in_channels, no_filters]
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))

# bias for each output
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

# add convolution step on input images
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

# add RELU layer
h_conv1 = tf.nn.relu(convolve1)
# add max pool layer
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

# weight/bias for conv2 layer
# shape = [filter_height, filter_weight, in_channels (32 output as input), no_filters]
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

# add second convolution step on the output of first convolution layer
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2

# add RELU for second conv step
h_conv2 = tf.nn.relu(convolve2)

# add max pool layer for second conv layer
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

# flatten the output of second conv layer
# 64 for each filter in second conv layer
# 7x7 as the size of reduced image
# 28x28 original image.. first pool (2x2) will reduce it to 14x14.. second pool (2x2) will reduce to 7x7
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

# create weights and bias for final layer (hidden layer size - 1024)
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs
fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1
# use the RELU activation function for this layer
h_fc1 = tf.nn.relu(fcl)

# Add drop output to the hidden layer
# some nodes will randomly switched off during each training step
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

# Last output layer.. 1024 inputs from hidden layer and 10 output
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

fc=tf.matmul(layer_drop, W_fc2) + b_fc2
y_CNN= tf.nn.softmax(fc)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
# Use Adams optimizer to minimize error between actual and derived y values
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define prediction & accuracy
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

session.close()
