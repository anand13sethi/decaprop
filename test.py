import numpy as np
import tensorflow as tf

# input_vec = tf.zeros([1, 200, 1])
# # y = np.tile(x, [1, 2, 1])
# # input_vec = tf.reshape(input_vec, [-1, 3])
# input_vec = tf.concat(input_vec, 2)
x = tf.zeros((32, 2, 3))
y = tf.zeros((32, 1, 5))
z = tf.concat([x, y], 1)
print(z.shape)
# a = z[:z.shape[0]//2, :, :]
# b = z[z.shape[0]//2:, :, :]
# print(input_vec.shape)
# print(a.shape, b.shape)