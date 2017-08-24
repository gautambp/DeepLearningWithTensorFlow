# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read mnist (handwriting) images
# each image is 28x28 = 784 pixels
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# CNN Architecture
# 784 inputs (for each image pixel)
# 10 outputs (1 for each digit)

session = tf.Session()

# create placeholders for inputs and outputs
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# create variables for weights 
# since input nodes are 784 and output nodes are 10, we need array(784, 10)
w = tf.Variable(tf.zeros([784, 10], tf.float32))

# create variables for bias
# we need 10 to feed into each 10 output
b = tf.Variable(tf.zeros([10], tf.float32))

# initialize variables
session.run(tf.global_variables_initializer())

# apply weight to input and add bias
u = tf.matmul(x, w) + b

# we'll be using softmax activation function to get the output - y
y = tf.nn.softmax(u)

# cost functon that utilizes actual y_ with computed y
# since softmax function gives exp value of y.. we need to compare y_ with log(y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# we'll be using gradient optimizer to minimize the cost/entropy
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

# do the actual training
for i in range(1000):
    # read the batch and train
    print("Processing training iteration - {}/1000".format(i))
    batch = mnist.train.next_batch(50)
    session.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

# now test the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

session.close()
