import numpy as np
import tensorflow as tf

# input_vec = tf.zeros([1, 200, 1])
# # y = np.tile(x, [1, 2, 1])
# # input_vec = tf.reshape(input_vec, [-1, 3])
# input_vec = tf.concat(input_vec, 2)
t = []
t1 = [[1], [4]]
t2 = [[1], [4]]
t.append(t1)
t.append(t2)
x = tf.concat(t, 1)
# print(input_vec.shape)
print(x.shape)