import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.05
training_epochs = 50
batch_size = 1000
display_step = 1

input_nodes = 784
hidden_nodes = 300
output_nodes = 10

# tf Graph Input
x = tf.placeholder(tf.float32, [None, input_nodes])
y = tf.placeholder(tf.float32, [None, output_nodes])

# Set model weights
W1 = tf.Variable(tf.random_uniform([input_nodes, output_nodes]))
b1 = tf.Variable(tf.random_uniform([output_nodes]))

# Construct model
output = tf.matmul(x, W1) + b1
pred = tf.nn.softmax(output)

# Minimize error using cross entropy
# cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(pred)))
cost = tf.losses.log_loss(y, pred)
grad_W1, grad_b1 = tf.gradients(xs=[W1, b1], ys=cost)

W1 = W1.assign(W1 - learning_rate * grad_W1)
b1 = b1.assign(b1 - learning_rate * grad_b1)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            nW1, nb1, c = sess.run([W1, b1, cost], feed_dict={x: batch_xs, y: batch_ys})

            # print(nW1, nb1, nW2, nb2, c)
            # print(nW2)

            # Compute average loss
            avg_cost += c

        avg_cost /= total_batch

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    print(len(filter(lambda t: t, correct_prediction.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
