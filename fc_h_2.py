from datetime import timedelta

import tensorflow as tf
import matplotlib.pyplot as plt

import time
import os

from utils import variable_summaries, print_results, load_data

"""
    Same as homework 1, but add nesterov and adam optimizers
"""


def create_network(img_size, num_channels, num_classes, num_fc_layer1_output, learning_rate, mu, momentum):

    input_values = img_size*img_size*num_channels

    # PLACEHOLDER VARIABLES
    x = tf.placeholder(tf.float32, shape=[None, input_values], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    fc_layer1_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    mu = tf.placeholder(tf.float32, name="mu")

    iteration = tf.placeholder(tf.float32, name="iteration")

    placeholders = {
        'x': x,
        'y_true': y_true,
        'y_true_cls': y_true_cls,
        'fc_layer1_keep_prob': fc_layer1_keep_prob,
        'phase_train': phase_train,
        'mu': mu,
        'iteration': iteration
    }

    batch_size = tf.shape(x)[0]

    grad_W1 = tf.constant(0.0)
    grad_b1 = tf.constant(0.0)
    grad_W2 = tf.constant(0.0)
    grad_b2 = tf.constant(0.0)

    grad_W1_momentum = tf.constant(0.0)
    grad_b1_momentum = tf.constant(0.0)
    grad_W2_momentum = tf.constant(0.0)
    grad_b2_momentum = tf.constant(0.0)

    grad_W1_momentum_prev = tf.constant(0.0)
    grad_b1_momentum_prev = tf.constant(0.0)
    grad_W2_momentum_prev = tf.constant(0.0)
    grad_b2_momentum_prev = tf.constant(0.0)

    grad_W2_m = tf.constant(1.0)
    grad_b2_m = tf.constant(1.0)
    grad_W1_m = tf.constant(1.0)
    grad_b1_m = tf.constant(1.0)

    # FULLY CONNECTED LAYER 1
    with tf.variable_scope('fc_1'):
        fc_weights1 = tf.Variable(tf.truncated_normal(shape=[input_values, num_fc_layer1_output], stddev=0.05))
        variable_summaries(fc_weights1)
        print(batch_size)
        fc_biases1 = tf.Variable(initial_value=tf.random_uniform(shape=[num_fc_layer1_output]), validate_shape=False)
        variable_summaries(fc_biases1)

        fc_layer1 = tf.matmul(x, fc_weights1) + fc_biases1
        tf.summary.histogram('fc_layer1', fc_layer1)

    # TANH
    with tf.variable_scope('fc_relu_1'):
        fc_layer1 = tf.nn.tanh(fc_layer1)

    # FULLY CONNECTED LAYER 2
    with tf.variable_scope('fc_2'):
        fc_weights2 = tf.Variable(tf.truncated_normal(shape=[num_fc_layer1_output, num_classes], stddev=0.05))
        variable_summaries(fc_weights2)

        fc_biases2 = tf.Variable(tf.random_uniform(shape=[num_classes]), validate_shape=False)
        variable_summaries(fc_biases2)

        fc_layer2 = tf.matmul(fc_layer1, fc_weights2) + fc_biases2
        tf.summary.histogram('fc_layer2', fc_layer2)

    # SOFTMAX
    with tf.variable_scope('softmax'):
        y_pred = tf.nn.softmax(fc_layer2)
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        tf.summary.histogram('y_pred', y_pred)

    # COST FUNCTION
    with tf.variable_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer2, labels=y_true))
        # cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
        # cost = tf.losses.log_loss(y_true, y_pred)
        tf.summary.histogram('cost', cost)

    # GRADIENT DESCENT METHOD

    # train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # grad_W2, grad_b2 = tf.gradients(ys=cost, xs=[fc_weights2, fc_biases2])
    # grad_W1, grad_b1 = tf.gradients(ys=cost, xs=[fc_weights1, fc_biases1])

    if momentum == "none":
        # No momentum
        sigmaprime = tf.multiply(tf.constant(1.0) - y_pred, y_pred)
        delta3 = tf.multiply(y_pred - y_true, sigmaprime)
        grad_W2 = tf.matmul(tf.transpose(fc_layer1), delta3)
        grad_b2 = tf.reduce_mean(delta3, reduction_indices=0)

        tanhprime = tf.constant(1.0) - tf.multiply(fc_layer1, fc_layer1)
        delta2 = tf.multiply(tf.matmul(delta3, tf.transpose(fc_weights2)), tanhprime)
        grad_W1 = tf.matmul(tf.transpose(x), delta2)
        grad_b1 = tf.reduce_mean(delta2, reduction_indices=0)

        fc_weights2 = fc_weights2.assign_sub(learning_rate * grad_W2)
        fc_biases2 = fc_biases2.assign_sub(learning_rate * grad_b2)

        fc_weights1 = fc_weights1.assign_sub(learning_rate * grad_W1)
        fc_biases1 = fc_biases1.assign_sub(learning_rate * grad_b1)

        train_step = [fc_weights2, fc_biases2, fc_weights1, fc_biases1]
    elif momentum == "classical":
        # Classical Momentum
        sigmaprime = tf.multiply(tf.constant(1.0) - y_pred, y_pred)
        delta3 = tf.multiply(y_pred - y_true, sigmaprime)
        grad_W2 = tf.matmul(tf.transpose(fc_layer1), delta3)
        grad_W2_momentum = mu * grad_W2_momentum + grad_W2
        grad_b2 = delta3
        grad_b2_momentum = tf.reduce_mean(mu * grad_b2_momentum + grad_b2, reduction_indices=0)

        tanhprime = tf.constant(1.0) - tf.multiply(fc_layer1, fc_layer1)
        delta2 = tf.multiply(tf.matmul(delta3, tf.transpose(fc_weights2)), tanhprime)
        grad_W1 = tf.matmul(tf.transpose(x), delta2)
        grad_W1_momentum = mu * grad_W1_momentum + grad_W1
        grad_b1 = delta2
        grad_b1_momentum = tf.reduce_mean(mu * grad_b1_momentum + grad_b1, reduction_indices=0)

        fc_weights2 = fc_weights2.assign_sub(learning_rate * grad_W2_momentum)
        fc_biases2 = fc_biases2.assign_sub(learning_rate * grad_b2_momentum)

        fc_weights1 = fc_weights1.assign_sub(learning_rate * grad_W1_momentum)
        fc_biases1 = fc_biases1.assign_sub(learning_rate * grad_b1_momentum)

        train_step = [fc_weights2, fc_biases2, fc_weights1, fc_biases1]
    elif momentum == "nesterov":
        # Nesterov Momentum

        # Tensorflow provided method
        momentum = tf.Variable(initial_value=0, name="momentum")
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(cost)

        # Approximation
        # sigmaprime = tf.multiply(tf.constant(1.0) - y_pred, y_pred)
        # delta3 = tf.multiply(y_pred - y_true, sigmaprime)
        #
        # # W2
        # grad_W2 = tf.matmul(tf.transpose(fc_layer1), delta3)
        # grad_W2_momentum_prev = grad_W2_momentum
        # grad_W2_momentum = mu * grad_W2_momentum - learning_rate * grad_W2
        # fc_weights2 = fc_weights2.assign_add((1 + mu) * grad_W2_momentum - mu * grad_W2_momentum_prev)
        #
        # # b2
        # grad_b2 = delta3
        # grad_b2_momentum_prev = grad_b2_momentum
        # grad_b2_momentum = tf.reduce_mean(mu * grad_b2_momentum - learning_rate * grad_b2, reduction_indices=0)
        # fc_biases2 = fc_biases2.assign_add((1 + mu) * grad_b2_momentum - mu * grad_b2_momentum_prev)
        #
        # tanhprime = tf.constant(1.0) - tf.multiply(fc_layer1, fc_layer1)
        # delta2 = tf.multiply(tf.matmul(delta3, tf.transpose(fc_weights2)), tanhprime)
        #
        # # W1
        # grad_W1 = tf.matmul(tf.transpose(x), delta2)
        # grad_W1_momentum_prev = grad_W1_momentum
        # grad_W1_momentum = mu * grad_W1_momentum - learning_rate * grad_W1
        # fc_weights1 = fc_weights1.assign_add((1 + mu) * grad_W1_momentum - mu * grad_W1_momentum_prev)
        #
        # # b1
        # grad_b1 = delta2
        # grad_b1_momentum_prev = grad_b1_momentum
        # grad_b1_momentum = tf.reduce_mean(mu * grad_b1_momentum - learning_rate * grad_b1, reduction_indices=0)
        # fc_biases1 = fc_biases1.assign_add((1 + mu) * grad_b1_momentum - mu * grad_b1_momentum_prev)

        # # Original formulation
        # sigmaprime = tf.multiply(tf.constant(1.0) - y_pred, y_pred)
        # delta3 = tf.multiply(y_pred - y_true, sigmaprime)
        #
        # # W2
        # grad_W2 = tf.matmul(tf.transpose(fc_layer1), delta3)
        # grad_W2_ahead = tf.add(grad_W2, tf.multiply(mu, grad_W2_momentum))
        # grad_W2_momentum = mu * grad_W2_momentum - learning_rate * grad_W2_ahead
        # fc_weights2 = fc_weights2.assign_add(grad_W2_momentum)
        #
        # # b2
        # grad_b2 = delta3
        # grad_b2_ahead = tf.add(grad_b2, tf.multiply(mu, grad_b2_momentum))
        # grad_b2_momentum = tf.reduce_mean(mu * grad_b2_momentum - learning_rate * grad_b2_ahead, reduction_indices=0)
        # fc_biases2 = fc_biases2.assign_add(grad_b2_momentum)
        #
        # tanhprime = tf.constant(1.0) - tf.multiply(fc_layer1, fc_layer1)
        # delta2 = tf.multiply(tf.matmul(delta3, tf.transpose(fc_weights2)), tanhprime)
        #
        # # W1
        # grad_W1 = tf.matmul(tf.transpose(x), delta2)
        # grad_W1_ahead = tf.add(grad_W1, tf.multiply(mu, grad_W1_momentum))
        # grad_W1_momentum = mu * grad_W1_momentum - learning_rate * grad_W1_ahead
        # fc_weights1 = fc_weights1.assign_add(grad_W1_momentum)
        #
        # # b1
        # grad_b1 = delta2
        # grad_b1_ahead = tf.add(grad_b1, tf.multiply(mu, grad_b1_momentum))
        # grad_b1_momentum = tf.reduce_mean(mu * grad_b1_momentum - learning_rate * grad_b1_ahead, reduction_indices=0)
        # fc_biases1 = fc_biases1.assign_add(grad_b1_momentum)
        #
        # train_step = [fc_weights2, fc_biases2, fc_weights1, fc_biases1]
    elif momentum == "adam":
        # Adam Momentum
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # beta1 = tf.constant(0.9)
        # beta2 = tf.constant(0.999)
        # eps = tf.constant(1e-8)
        #
        # sigmaprime = tf.multiply(tf.constant(1.0) - y_pred, y_pred)
        # delta3 = tf.multiply(y_pred - y_true, sigmaprime)
        #
        # # W2
        # grad_W2 = tf.matmul(tf.transpose(fc_layer1), delta3)
        # # fc_weights2 = fc_weights2.assign_sub(learning_rate * grad_W2)
        #
        # grad_W2_m = tf.multiply(beta1, grad_W2_m) + tf.multiply((tf.constant(1.0) - beta1), grad_W2)
        # grad_W2_momentum = tf.multiply(beta2, grad_W2_momentum) + tf.multiply((tf.constant(1.0) - beta2), (tf.pow(grad_W2, tf.constant(2.0))))
        # fc_weights2 = fc_weights2.assign_add(learning_rate * grad_W2_m / (tf.sqrt(grad_W2_momentum) + eps))
        #
        # # b2
        # grad_b2 = tf.reduce_mean(delta3, reduction_indices=0)
        # fc_biases2 = fc_biases2.assign_add(learning_rate * grad_b2)
        #
        # # grad_b2_m = beta1 * grad_b2_m + (1 - beta1) * grad_b2
        # # grad_b2_m_t = grad_b2_m / (1 - beta1 ** iteration)
        # # grad_b2_momentum = beta2 / grad_b2_momentum + (1 - beta2) * (grad_b2 ** 2)
        # # grad_b2_momentum_t = grad_b2_momentum / (1 - beta2 ** 2)
        # # fc_biases2 = fc_biases2.assign_sub(tf.reduce_mean(learning_rate * grad_b2_m_t / (tf.sqrt(grad_b2_momentum_t) + eps), reduction_indices=0))
        #
        # tanhprime = tf.constant(1.0) - tf.multiply(fc_layer1, fc_layer1)
        # delta2 = tf.multiply(tf.matmul(delta3, tf.transpose(fc_weights2)), tanhprime)
        #
        # # W1
        # grad_W1 = tf.matmul(tf.transpose(x), delta2)
        # fc_weights1 = fc_weights1.assign_add(learning_rate * grad_W1)
        #
        # # grad_W1_m = tf.multiply(beta1, grad_W1_m) + tf.multiply((tf.constant(1.0) - beta1), grad_W1)
        # # grad_W1_m_t = tf.divide(grad_W1_m, (tf.constant(1.0) - tf.pow(beta1, iteration)))
        # # grad_W1_momentum = tf.divide(beta2, grad_W1_momentum) + tf.multiply((tf.constant(1.0) - beta2), (tf.pow(grad_W1, tf.constant(2.0))))
        # # grad_W1_momentum_t = tf.divide(grad_W1_momentum, (tf.constant(1.0) - tf.pow(beta2, tf.constant(2.0))))
        # # fc_weights1 = fc_weights1.assign_sub(learning_rate * tf.divide(grad_W1_m_t, (tf.sqrt(grad_W1_momentum_t) + eps)))
        #
        # # b1
        # grad_b1 = tf.reduce_mean(delta2, reduction_indices=0)
        # fc_biases1 = fc_biases1.assign_add(learning_rate * grad_b1)
        #
        # # grad_b1_m = beta1 * grad_b1_m + (1 - beta1) * grad_b1
        # # grad_b1_m_t = grad_b1_m / (1 - beta1 ** iteration)
        # # grad_b1_momentum = beta2 / grad_b1_momentum + (1 - beta2) * (grad_b1 ** 2)
        # # grad_b1_momentum_t = grad_b1_momentum / (1 - beta2 ** 2)
        # # fc_biases1 = fc_biases1.assign_sub(tf.reduce_mean(learning_rate * grad_b1_m_t / (tf.sqrt(grad_b1_momentum_t) + eps), reduction_indices=0))
        #
        # train_step = [fc_weights2, fc_biases2, fc_weights1, fc_biases1]
    else:
        # No momentum
        grad_W2, grad_b2 = tf.gradients(ys=cost, xs=[fc_weights2, fc_biases2])
        grad_W1, grad_b1 = tf.gradients(ys=cost, xs=[fc_weights1, fc_biases1])

        fc_weights2 = fc_weights2.assign_sub(learning_rate * grad_W2)
        fc_biases2 = fc_biases2.assign_sub(learning_rate * grad_b2)

        fc_weights1 = fc_weights1.assign_sub(learning_rate * grad_W1)
        fc_biases1 = fc_biases1.assign_sub(learning_rate * grad_b1)

        train_step = [fc_weights2, fc_biases2, fc_weights1, fc_biases1]


    # PERFORMANCE MEASURES
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram('accuracy', accuracy)

    return train_step, cost, accuracy, y_pred_cls, y_true_cls, placeholders


def train_network(data,
                  train_step,
                  cost,
                  accuracy,
                  num_iterations,
                  train_batch_size,
                  placeholders,
                  saver,
                  save_dir,
                  plot_dir,
                  log_dir,
                  display_step,
                  mu):
    with tf.Session() as session:
        print("Session opened!")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, session.graph)
        session.run(tf.global_variables_initializer())
        print("Global initializer run!")

        total_iterations = 0
        start_time = time.time()
        training_acc = []
        training_cost = []
        testing_acc = []
        mu_step = (0.99 - mu)/num_iterations
        for i in range(total_iterations,
                       total_iterations + num_iterations):
            mu += mu_step

            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            feed_dict_train = {placeholders['x']: x_batch,
                               placeholders['y_true']: y_true_batch,
                               placeholders['phase_train']: True,
                               placeholders['mu']: mu,
                               placeholders['iteration']: i
                               }

            # train_step.run(feed_dict_train)
            # summary, _, _, _, _ = session.run([merged, train_step[0], train_step[1], train_step[2], train_step[3]], feed_dict=feed_dict_train)
            session.run([train_step], feed_dict=feed_dict_train)
            # train_writer.add_summary(summary, i)

            if i % display_step == 0:
                # acc = session.run(accuracy, feed_dict=feed_dict_train)
                feed_dict_cross_train = {placeholders['x']: x_batch,
                                         placeholders['y_true']: y_true_batch,
                                         placeholders['phase_train']: False,
                                         placeholders['mu']: mu,
                                         placeholders['iteration']: i
                                         }
                train_acc = accuracy.eval(feed_dict_cross_train)
                training_acc += [(i+1, train_acc)]

                c = cost.eval(feed_dict_cross_train)
                training_cost += [(i+1, c)]

                x_batch, y_true_batch = data.test.next_batch(train_batch_size)

                feed_dict_cross_test = {placeholders['x']: x_batch,
                                        placeholders['y_true']: y_true_batch,
                                        placeholders['phase_train']: False,
                                        placeholders['mu']: mu,
                                        placeholders['iteration']: i
                                        }
                test_acc = accuracy.eval(feed_dict_cross_test)
                testing_acc += [(i + 1, test_acc)]

                msg = "Iteration: {0:>6}, Loss: {1:>6.1%}, Train accuracy: {2:>6.1%}, Test accuracy: {3:>6.1%}"
                print(msg.format(i, c, train_acc, test_acc))

        total_iterations += num_iterations

        end_time = time.time()
        time_dif = end_time - start_time

        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        saver.save(sess=session, save_path=save_dir)

        plt.plot(*zip(*training_acc))
        plt.savefig(plot_dir + "train_accuracy.png")
        plt.clf()

        plt.plot(*zip(*testing_acc))
        plt.savefig(plot_dir + "test_accuracy.png")
        plt.clf()

        plt.plot(*zip(*training_cost))
        plt.savefig(plot_dir + "cost.png")
        plt.clf()

        train_writer.close()

        return training_acc, training_cost, testing_acc


def test_network(test_batch_size, placeholders, saver, save_dir, accuracy, y_true_cls, y_pred_cls, data):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver.restore(sess=session, save_path=save_dir)

        num_test = len(data.test.images)
        num_iterations = int(num_test / test_batch_size) + 1

        i = 0
        avg_acc = 0
        y_pred = []
        y_true = []
        while i < num_iterations:
            x_batch, y_batch = data.test.next_batch(test_batch_size)

            feed_dict_test = {placeholders['x']: x_batch,
                              placeholders['y_true']: y_batch,
                              placeholders['phase_train']: False}

            batch_y_pred, batch_y_true, batch_acc = session.run([y_pred_cls, y_true_cls, accuracy], feed_dict=feed_dict_test)
            y_true += batch_y_true.T.tolist()
            y_pred += batch_y_pred.T.tolist()
            avg_acc += batch_acc

            i += 1

        acc = float(avg_acc) / num_iterations

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%}"
        print(msg.format(acc))

        return y_true, y_pred, acc


def run_experiment(params):
    tf.reset_default_graph()

    train_step, \
    cost, \
    accuracy, \
    y_pred_cls, \
    y_true_cls, \
    placeholders = create_network(
                                params['img_size'],
                                params['num_channels'],
                                params['num_classes'],
                                params['num_fc_layer1_output'],
                                params['learning_rate'],
                                params['mu'],
                                params['momentum'])

    saver = tf.train.Saver()
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    (train_acc, cost, test_acc) = train_network(params['data'],
                                                train_step,
                                                cost,
                                                accuracy,
                                                params['num_iterations'],
                                                params['train_batch_size'],
                                                placeholders,
                                                saver,
                                                params['save_dir'],
                                                params['plot_dir'],
                                                params['log_dir'],
                                                params['display_step'],
                                                params['mu'])

    cls_true, cls_pred, acc = test_network(params['test_batch_size'],
                                      placeholders,
                                      saver,
                                      params['save_dir'],
                                      accuracy,
                                      y_pred_cls,
                                      y_true_cls,
                                      params['data'])

    print_results(cls_pred,
                  cls_true,
                  "",
                  params['plot_dir'])

    return train_acc, cost, test_acc, acc


if __name__ == "__main__":
    sgd_types = ['manual', 'gradients', 'optimizer']

    datasets = [('mnist', 28, 1, 5000, 100, 1e-3, 100), ('cifar10', 32, 3, 10000, 100, 1e-3, 100)]
    base_dir = os.getcwd() + "/data/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # momentums = ["none", "classical", "nesterov", "adam"]
    momentums = ["none", "classical", "nesterov", "adam"]

    for dataset, img_size, num_channels, num_iterations, y_lim, learning_rate, display_step in datasets:
        dataset_dir = base_dir + dataset
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        data_dir = dataset_dir + "/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        all_momentums_test = []
        all_momentums_train = []
        for momentum in momentums:

            save_dir = dataset_dir + "/checkpoints_1_" + str(learning_rate) + "_" + str(num_iterations) + "_" + momentum + "/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plot_dir = dataset_dir + "/plots_1_" + str(learning_rate) + "_" + str(num_iterations) + "_" + momentum + "/"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            log_dir = dataset_dir + "/logs_1_" + str(learning_rate) + "_" + str(num_iterations) + "_" + momentum + "/"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            num_classes = 10
            filter_size1 = 5
            filter_size2 = 5
            num_filters1 = 30
            num_filters2 = 60
            shape1 = [filter_size1, filter_size1, num_channels, num_filters1]
            shape2 = [filter_size2, filter_size2, num_filters1, num_filters2]
            num_fc_layer1_output = 400
            num_fc_layer2_output = num_classes

            train_batch_size = 500
            test_batch_size = 500

            mu = 0.3

            data = load_data(data_dir, dataset)

            params = {
                'img_size': img_size,
                'num_channels': num_channels,
                'num_classes': num_classes,
                'num_iterations': num_iterations,
                'shape1': shape1,
                'shape2': shape2,
                'num_fc_layer1_output': num_fc_layer1_output,
                'num_fc_layer2_output': num_fc_layer2_output,
                'learning_rate': learning_rate,
                'data': data,
                'save_dir': save_dir,
                'plot_dir': plot_dir,
                'log_dir': log_dir,
                'train_batch_size': train_batch_size,
                'test_batch_size': test_batch_size,
                'display_step': display_step,
                'mu': mu,
                'momentum': momentum
            }

            stats = run_experiment(params)
            plt.clf()

            plt.title(str(stats[3]))
            plt.plot(*zip(*stats[0]), label='train_accuracy')
            plt.plot(*zip(*stats[2]), label='test_accuracy')
            plt.plot(*zip(*stats[1]), label='cost')

            plt.legend(loc='best')
            plt.savefig(plot_dir + "all.png")

            all_momentums_train += [(stats[0], momentum + "+" + str(stats[3]))]
            all_momentums_test += [(stats[2], momentum + "+" + str(stats[3]))]

        plt.clf()
        for s in all_momentums_train:
            plt.plot(*zip(*s[0]), label=s[1])
        plt.legend(loc='best')
        plt.savefig(data_dir + "accuracy_train.png")

        plt.clf()
        for s in all_momentums_test:
            plt.plot(*zip(*s[0]), label=s[1])
        plt.legend(loc='best')
        plt.savefig(data_dir + "accuracy_test.png")
