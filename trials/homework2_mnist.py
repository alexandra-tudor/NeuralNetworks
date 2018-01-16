from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False)

# Training Parameters
learning_rate = 0.06
mu = 0.9
sigma = 0.1
num_steps = 100
batch_size = 1000
training_epochs = 5000
display_step = 100

# Network Parameters
input_nodes = 784
output_nodes = 10
dropout = 0.75

x = tf.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.placeholder(tf.int32, (None, 10))

conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
conv1_b = tf.Variable(tf.zeros(6))

conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
conv2_b = tf.Variable(tf.zeros(16))

fc1_W = tf.Variable(tf.truncated_normal(shape=(784, 120), mean=mu, stddev=sigma))
fc1_b = tf.Variable(tf.zeros(120))

fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
fc2_b = tf.Variable(tf.zeros(84))

fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
fc3_b = tf.Variable(tf.zeros(10))
"""
    Architecture
        Layer 1: Convolutional. The output shape should be 28x28x6.
            Activation.
        Pooling. The output shape should be 14x14x6.
        Layer 2: Convolutional. The output shape should be 10x10x16.
            Activation. 
        Pooling. The output shape should be 5x5x16.
        Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
        Layer 3: Fully Connected. This should have 120 outputs.
            Activation. 
        Layer 4: Fully Connected. This should have 84 outputs.
            Activation. 
        Layer 5: Fully Connected (Logits). This should have 10 outputs.
"""

conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
fc0 = flatten(conv2)
fc1 = tf.matmul(fc0, fc1_W) + fc1_b
fc1 = tf.nn.relu(fc1)
fc2 = tf.matmul(fc1, fc2_W) + fc2_b
fc2 = tf.nn.relu(fc2)
logits = tf.matmul(fc2, fc3_W) + fc3_b

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

plot_cost = []
plot_accuracy = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = mnist.train.num_examples

    print("Training...")
    print()

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_accuracy = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # # train
        # for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(training_operation, feed_dict={x: batch_xs, y: batch_ys})

        # # evaluate
        # for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        acc = sess.run(accuracy_operation, feed_dict={x: batch_xs, y: batch_ys})
        # avg_cost += loss
        # total_accuracy += (acc * batch_size)

        # avg_cost /= total_batch
        # validation_accuracy = total_accuracy / num_examples

        # plot_cost += [(epoch, loss)]
        # plot_accuracy += [(epoch, acc)]

        if (epoch + 1) % display_step == 0:
            # print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(loss))
            print("Validation Accuracy = {:.3f}".format(acc))

        # print()

# plt.scatter(*zip(*plot_cost))
# plt.savefig("h3_mnist_cost.png")
#
# plt.clf()
#
# plt.scatter(*zip(*plot_accuracy))
# plt.savefig("h3_mnist_acc.png")