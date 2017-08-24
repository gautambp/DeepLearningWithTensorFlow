# -*- coding: utf-8 -*-

import tensorflow as tf

#create different dimensional data structures (tensors)
scalar = tf.constant([1])
vector = tf.constant([1, 2])
matrix = tf.constant([[1, 2],[3, 4]])
tensor = tf.constant([ [[1, 2], [3, 4]], [[5, 6], [7, 8]]])

with tf.Session() as session:
    result = session.run(scalar)
    print("scalar result", result)
    result = session.run(vector)
    print("vector result", result)
    result = session.run(matrix)
    print("matrix result", result)
    result = session.run(tensor)
    print("tensor result", result)

# apply diff ops
m2 = tf.constant([[5, 6], [7, 8]])
mAdd = matrix + m2
mMult = matrix * m2

with tf.Session() as session:
    print("Matrix add : \n", session.run(mAdd))
    print("Matrix multiplicatio : \n", session.run(mMult))

# let's check variables
v1 = tf.Variable(0)
c1 = tf.constant(1)
new_v1 = tf.add(v1, c1)
update_v1 = tf.assign(v1, new_v1)

# variables require initialization before running
# so create an init op
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    print("v1 = ", session.run(v1))
    for _ in range(3):
        session.run(update_v1)
        print("v1 = ", session.run(v1))

# Placeholders are kind of variables that are populated/initialized at later time
a = tf.placeholder(tf.float32)
b = a * 2

with tf.Session() as session:
    # pass the value to the placeholder when you try to run
    result = session.run(b, feed_dict={a:3.5})
    print(result)
    # we can pass any tensor value for the placeholder.. so long as it is of type defined
    # in the placeholder (e.g, vector or matrix of float)
    result = session.run(b, feed_dict={a:[3, 4]})
    print(result)
    
