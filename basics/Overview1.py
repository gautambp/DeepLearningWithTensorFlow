# -*- coding: utf-8 -*-

import tensorflow as tf

# define data structure
a = tf.constant([2])
b = tf.constant([3])

# define operations
# you could also do c = a + b
c = tf.add(a, b)

# create session and evaluate c
session = tf.Session()
result = session.run(c)

print(result)

session.close()

# alternatively use with block
with tf.Session() as session:
    result = session.run(c)
    print(result)

