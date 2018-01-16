import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.00001
training_epochs = 400
batch_size = 3000
display_step = 1

input_nodes = 784
hidden_nodes = 300
output_nodes = 10

grad_W1 = tf.constant(0.0)
grad_b1 = tf.constant(0.0)

# tf Graph Input
x = tf.placeholder(tf.float32, [None, input_nodes])
y = tf.placeholder(tf.float32, [None, output_nodes])

# Set model weights
W1 = tf.Variable(tf.random_uniform([input_nodes, output_nodes]))
b1 = tf.Variable(tf.random_uniform([batch_size, output_nodes]), validate_shape=False)

print(x)
print(W1)
print (b1)
print(tf.matmul(x, W1))
# Construct model
output = tf.matmul(x, W1) + b1
pred = tf.nn.softmax(output)

# Minimize error using cross entropy
# cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(pred)))
cost = tf.losses.log_loss(y, pred)
# grad_W1, grad_b1 = tf.gradients(xs=[W1, b1], ys=cost)

sigmaprime = tf.multiply(tf.constant(1.0) - pred, pred)
delta3 = tf.multiply(pred - y, sigmaprime)
grad_W1 += tf.matmul(tf.transpose(x), delta3)
grad_b1 += delta3

W1 = W1.assign(W1 - learning_rate * grad_W1)
b1 = b1.assign(b1 - learning_rate * grad_b1)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            nW1, nb1, c = sess.run([W1, b1, cost], feed_dict={x: batch_xs, y: batch_ys})

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
    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
