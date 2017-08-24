# -*- coding: utf-8 -*-

import tensorflow as tf

# build an image 10x10
input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
# build filter/kernal.. 3x3
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))

#now define convolution op
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    
    print("Input \n {0} \n", format(input.eval()))
    print("Filter \n {0} \n", format(filter.eval()))
    print("Result with valid padding\n")
    print(session.run(op))
    print("Result with same padding\n")
    print(session.run(op2))

