from datetime import timedelta

import tensorflow as tf
import matplotlib.pyplot as plt

import time
import os

from utils import variable_summaries, print_results, load_data


def create_network(img_size, num_channels, num_classes, num_fc_layer1_output, learning_rate, bn=False):

    input_values = img_size*img_size*num_channels

    # PLACEHOLDER VARIABLES
    x = tf.placeholder(tf.float32, shape=[None, input_values], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    fc_layer1_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    placeholders = {
        'x': x,
        'y_true': y_true,
        'y_true_cls': y_true_cls,
        'fc_layer1_keep_prob': fc_layer1_keep_prob,
        'phase_train': phase_train
    }

    batch_size = tf.shape(x)[0]

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

    # DROPOUT LAYER 1
    with tf.variable_scope('dropout_1'):
        fc_layer1_dropout = tf.nn.dropout(fc_layer1, fc_layer1_keep_prob)

    # FULLY CONNECTED LAYER 2
    with tf.variable_scope('fc_2'):
        fc_weights2 = tf.Variable(tf.truncated_normal(shape=[num_fc_layer1_output, num_classes], stddev=0.05))
        variable_summaries(fc_weights2)

        fc_biases2 = tf.Variable(tf.random_uniform(shape=[num_classes]), validate_shape=False)
        variable_summaries(fc_biases2)

        fc_layer2 = tf.matmul(fc_layer1_dropout, fc_weights2) + fc_biases2
        tf.summary.histogram('fc_layer2', fc_layer2)

    # SOFTMAX
    with tf.variable_scope('softmax'):
        y_pred = tf.nn.softmax(fc_layer2)
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        tf.summary.histogram('y_pred', y_pred)

    # COST FUNCTION
    with tf.variable_scope('cost'):
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer2, labels=y_true))
        # cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
        cost = tf.losses.log_loss(y_true, y_pred)
        tf.summary.histogram('cost', cost)

    # GRADIENT DESCENT METHOD
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    sigmaprime = tf.multiply(tf.constant(1.0) - y_pred, y_pred)
    delta3 = tf.multiply(y_pred - y_true, sigmaprime)

    grad_W2 = tf.matmul(tf.transpose(fc_layer1), delta3)
    grad_b2 = tf.reduce_mean(delta3, reduction_indices=0)

    grad_W2, grad_b2 = tf.gradients(ys=cost, xs=[fc_weights2, fc_biases2])
    #
    # fc_weights2 = fc_weights2.assign_sub(learning_rate * grad_W2)
    # fc_biases2 = fc_biases2.assign_sub(learning_rate * grad_b2)

    # tanhprime = tf.constant(1.0) - tf.multiply(fc_layer1, fc_layer1)
    # delta2 = tf.multiply(tf.matmul(delta3, tf.transpose(fc_weights2)), tanhprime)
    #
    # grad_W1 = tf.matmul(tf.transpose(x), delta2)
    # grad_b1 = tf.reduce_mean(delta2, reduction_indices=0)

    # grad_W1, grad_b1 = tf.gradients(ys=cost, xs=[fc_weights1, fc_biases1])
    #
    # fc_weights1 = fc_weights1.assign_sub(learning_rate * grad_W1)
    # fc_biases1 = fc_biases1.assign_sub(learning_rate * grad_b1)

    # PERFORMANCE MEASURES
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram('accuracy', accuracy)

    # train_step = [fc_weights2, fc_biases2, fc_weights1, fc_biases1]
    return train_step, cost, accuracy, y_pred_cls, y_true_cls, placeholders


def train_network(data, train_step, cost, accuracy, num_iterations, train_batch_size, placeholders, saver, save_dir, plot_dir, log_dir, display_step):
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
        for i in range(total_iterations,
                       total_iterations + num_iterations):

            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            feed_dict_train = {placeholders['x']: x_batch,
                               placeholders['y_true']: y_true_batch,
                               placeholders['fc_layer1_keep_prob']: 0.5,
                               placeholders['phase_train']: True,
                               }

            # train_step.run(feed_dict_train)
            # summary, _, _, _, _ = session.run([merged, train_step[0], train_step[1], train_step[2], train_step[3]], feed_dict=feed_dict_train)
            a = session.run([train_step], feed_dict=feed_dict_train)
            # train_writer.add_summary(summary, i)

            if i % display_step == 0:
                # acc = session.run(accuracy, feed_dict=feed_dict_train)
                feed_dict_cross_train = {placeholders['x']: x_batch,
                                         placeholders['y_true']: y_true_batch,
                                         placeholders['fc_layer1_keep_prob']: 1.0,
                                         placeholders['phase_train']: False
                                         }
                train_acc = accuracy.eval(feed_dict_cross_train)
                training_acc += [(i+1, train_acc)]

                c = cost.eval(feed_dict_cross_train)
                training_cost += [(i+1, c)]

                x_batch, y_true_batch = data.test.next_batch(train_batch_size)

                feed_dict_cross_test = {placeholders['x']: x_batch,
                                        placeholders['y_true']: y_true_batch,
                                        placeholders['fc_layer1_keep_prob']: 1.0,
                                        placeholders['phase_train']: False
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
                              placeholders['fc_layer1_keep_prob']: 1.0,
                              placeholders['phase_train']: False
                              }

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
                                params['learning_rate'])

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
                                                params['display_step'])

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

    datasets = [('mnist', 28, 1, 30000, 100, 0.8, 500), ('cifar10', 32, 3, 50000, 100, 0.8, 500)]
    base_dir = os.getcwd() + "/data/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for dataset, img_size, num_channels, num_iterations, y_lim, learning_rate, display_step in datasets:
        dataset_dir = base_dir + dataset
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        data_dir = dataset_dir + "/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        save_dir = dataset_dir + "/checkpoints_1_" + str(learning_rate) + "_" + str(num_iterations) + "_dropout/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plot_dir = dataset_dir + "/plots_1_" + str(learning_rate) + "_" + str(num_iterations) + "_dropout/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        log_dir = dataset_dir + "/logs_1_" + str(learning_rate) + "_" + str(num_iterations) + "_dropout/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        num_classes = 10
        num_fc_layer1_output = 400
        num_fc_layer2_output = num_classes

        train_batch_size = 1000
        test_batch_size = 1000

        data = load_data(data_dir, dataset)

        params = {
            'img_size': img_size,
            'num_channels': num_channels,
            'num_classes': num_classes,
            'num_iterations': num_iterations,
            'num_fc_layer1_output': num_fc_layer1_output,
            'learning_rate': learning_rate,
            'data': data,
            'save_dir': save_dir,
            'plot_dir': plot_dir,
            'log_dir': log_dir,
            'train_batch_size': train_batch_size,
            'test_batch_size': test_batch_size,
            'display_step': display_step
        }

        stats = run_experiment(params)
        plt.clf()

        plt.title(str(stats[3]) + "_dropout")
        plt.plot(*zip(*stats[0]), label='train_accuracy')
        plt.plot(*zip(*stats[2]), label='test_accuracy')
        plt.plot(*zip(*stats[1]), label='cost')

        plt.legend(loc='best')
        plt.savefig(plot_dir + "all.png")
