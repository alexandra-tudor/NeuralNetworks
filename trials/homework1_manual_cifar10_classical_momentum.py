import tensorflow as tf
import cifar10
from sklearn.utils import shuffle

# Parameters
from utils import random_batch

learning_rate = 0.001
mu = 0.9

training_epochs = 50
batch_size = 300
display_step = 1

input_nodes = 3072
hidden_nodes = 1000
output_nodes = 10

cifar10.data_path = "/home/ardelaxela/Research/projects/NeuralNetworks/data/"
cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

images_train = images_train.reshape([len(images_train), input_nodes])
images_test = images_test.reshape([len(images_test), input_nodes])

images_train, labels_train = shuffle(images_train, labels_train)
images_test, labels_test = shuffle(images_test, labels_test)

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# tf Graph Input
x = tf.placeholder(tf.float32, [None, input_nodes])
y = tf.placeholder(tf.float32, [None, output_nodes])

# Set model weights
W1 = tf.Variable(tf.random_uniform([input_nodes, hidden_nodes]))
b1 = tf.Variable(tf.random_uniform([batch_size, hidden_nodes]))

W2 = tf.Variable(tf.random_uniform([hidden_nodes, output_nodes]))
b2 = tf.Variable(tf.random_uniform([batch_size, output_nodes]))

# Construct model
hidden = tf.matmul(x, W1) + b1
a2 = tf.tanh(hidden)
output = tf.matmul(a2, W2) + b2
pred = tf.nn.softmax(output)

grad_W1 = tf.constant(0.0)
grad_b1 = tf.constant(0.0)
grad_W2 = tf.constant(0.0)
grad_b2 = tf.constant(0.0)

grad_W1_momentum = tf.constant(0.0)
grad_b1_momentum = tf.constant(0.0)
grad_W2_momentum = tf.constant(0.0)
grad_b2_momentum = tf.constant(0.0)

# Minimize error using cross entropy
# cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(pred)))
cost = tf.losses.log_loss(y, pred)

# grad_W2 += tf.gradients(xs=W2, ys=cost)
# grad_b2 += tf.gradients(xs=b2, ys=cost)
sigmaprime = tf.multiply(tf.constant(1.0) - pred, pred)
delta3 = tf.multiply(pred - y, sigmaprime)

grad_W2 += tf.matmul(tf.transpose(a2), delta3)
grad_W2_momentum = mu * grad_W2_momentum + grad_W2

grad_b2 += delta3
grad_b2_momentum = mu * grad_b2_momentum + grad_b2

W2 = W2.assign(W2 - learning_rate * grad_W2_momentum)
b2 = b2.assign(b2 - learning_rate * grad_b2_momentum)

# grad_W1 += tf.gradients(xs=[W1], ys=cost)
# grad_b1 += tf.gradients(xs=[b1], ys=cost)
tanhprime = tf.constant(1.0) - tf.multiply(a2, a2)
delta2 = tf.multiply(tf.matmul(delta3, tf.transpose(W2)), tanhprime)

grad_W1 += tf.matmul(tf.transpose(x), delta2)
grad_W1_momentum = mu * grad_W1_momentum + grad_W1

grad_b1 += delta2
grad_b1_momentum = mu * grad_b1_momentum + grad_b1

W1 = W1.assign(W1 - learning_rate * grad_W1_momentum)
b1 = b1.assign(b1 - learning_rate * grad_b1_momentum)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: images_test[:300], y: labels_test[:300]}))

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(images_train) / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = random_batch(images_train, labels_train, 300)
            # print(tanhprime.eval(feed_dict={x: batch_xs, y: batch_ys}))

            # Fit training using batch data
            nW1, nb1, nW2, nb2, c, gW2 = sess.run([W1, b1, W2, b2, cost, grad_W2], feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += c

        avg_cost /= total_batch

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: images_test[:300], y: labels_test[:300]}))
