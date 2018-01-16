import os

import numpy as np
import tensorflow as tf

import mnist
from utils import dense_to_one_hot, sigmaprime, sigma

data_dir = 'data'
cifar10_path = os.path.join(data_dir, 'cifar10')
mnist_path = os.path.join(data_dir, 'mnist')

# check for existence
os.path.exists(data_dir)

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

# number of neurons in each layer
input_num_units = 28*28
hidden_num_units = 300
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

fully_connected_layer1 = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
fully_connected_layer1_tanh = sigma(fully_connected_layer1)
# fully_connected_layer1_tanh = tf.nn.tanh(fully_connected_layer1)
# fully_connected_layer1_tanh = fully_connected_layer1

fully_connected_layer2 = tf.add(tf.matmul(fully_connected_layer1_tanh, weights['output']), biases['output'])
fully_connected_layer2_tanh = sigma(fully_connected_layer2)
# fully_connected_layer2_tanh = tf.nn.tanh(fully_connected_layer2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected_layer2_tanh, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

d_z_2 = tf.multiply(loss, sigmaprime(fully_connected_layer2))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(fully_connected_layer1_tanh), d_z_2)

d_a_1 = tf.matmul(d_z_2, tf.transpose(weights['output']))
d_z_1 = tf.multiply(d_a_1, sigmaprime(fully_connected_layer1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(x), d_z_1)

eta = tf.constant(0.5)
step = [
    tf.assign(weights['hidden'], tf.subtract(weights['hidden'], tf.multiply(eta, d_w_1))),
    tf.assign(biases['hidden'], tf.subtract(biases['hidden'], tf.multiply(eta, tf.reduce_mean(d_b_1, axis=[0])))),
    tf.assign(weights['output'], tf.subtract(weights['output'], tf.multiply(eta, d_w_2))),
    tf.assign(biases['output'], tf.subtract(biases['output'], tf.multiply(eta, tf.reduce_mean(d_b_2, axis=[0]))))
]

acct_mat = tf.equal(tf.argmax(fully_connected_layer2_tanh, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

# load data
mnist = mnist.load_mnist(mnist_path)

# cifar10_train_images, \
# cifar10_train_labels, \
# cifar10_validation_images, \
# cifar10_validation_labels, \
# cifar10_test_images, \
# cifar10_test_labels = cifar10.maybe_download_and_extract(cifar10_path)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    batch_ys = dense_to_one_hot(batch_ys, 10)

    sess.run(step, feed_dict={x: batch_xs,
                                y: batch_ys})
    if i % 1000 == 0:
        mnist_labels = dense_to_one_hot(mnist.test.labels[:1000], 10)
        res = sess.run(acct_res, feed_dict =
                       {x: mnist.test.images[:1000],
                        y: mnist_labels})
        print (i, res)
