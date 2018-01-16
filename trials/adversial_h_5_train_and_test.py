from datetime import timedelta

import tensorflow as tf
import time
import os
import itertools


from utils import variable_summaries, plot_confusion_matrix, plot_images, load_data


def batch_norm(x, phase_train):
    beta = tf.Variable(tf.constant(0.0, shape=[x.get_shape()[-1]]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[x.get_shape()[-1]]), name='gamma', trainable=True)
    axes = range(len(x.shape)-1)
    batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed


def create_network(img_size, num_channels, num_classes, shape1, shape2, num_fc_layer1_output, num_fc_layer2_output, learning_rate, bn=False):

    # PLACEHOLDER VARIABLES
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size * num_channels], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    fc_layer1_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    placeholders = {
        'x': x,
        'x_image': x_image,
        'y_true': y_true,
        'y_true_cls': y_true_cls,
        'fc_layer1_keep_prob': fc_layer1_keep_prob,
        'phase_train': phase_train
    }

    # CONVOLUTIONAL LAYER 1
    with tf.variable_scope('conv_1'):
        conv_weights1 = tf.Variable(tf.truncated_normal(shape1, stddev=0.05))
        conv_biases1 = tf.Variable(tf.constant(0.05, shape=[shape1[3]]))

        conv_layer1 = tf.nn.conv2d(input=x_image, filter=conv_weights1, strides=[1, 1, 1, 1], padding='SAME') + conv_biases1

    # POOLING LAYER 1
    with tf.variable_scope('pool_1'):
        conv_layer1 = tf.nn.max_pool(value=conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('bn_1'):
            conv_layer1 = batch_norm(conv_layer1, phase_train)

    # RELU
    with tf.variable_scope('relu_1'):
        conv_layer1 = tf.nn.relu(conv_layer1)

    # CONVOLUTIONAL LAYER 2
    with tf.variable_scope('conv_2'):
        conv_weights2 = tf.Variable(tf.truncated_normal(shape2, stddev=0.05))
        conv_biases2 = tf.Variable(tf.constant(0.05, shape=[shape2[3]]))

        conv_layer2 = tf.nn.conv2d(input=conv_layer1, filter=conv_weights2, strides=[1, 1, 1, 1], padding='SAME') + conv_biases2

    # POOLING LAYER 2
    with tf.variable_scope('pool_2'):
        conv_layer2 = tf.nn.max_pool(value=conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('bn_2'):
            conv_layer2 = batch_norm(conv_layer2, phase_train)

    # RELU
    with tf.variable_scope('relu_2'):
        conv_layer2 = tf.nn.relu(conv_layer2)

    # FLATTEN LAYER
    with tf.variable_scope('flatten'):
        layer_shape = conv_layer2.get_shape()
        num_features = layer_shape[1:4].num_elements()  # [num_images, img_height * img_width * num_channels]

        layer_flat = tf.reshape(conv_layer2, [-1, num_features])

    # FULLY CONNECTED LAYER 1
    with tf.variable_scope('fc_1'):
        fc_weights1 = tf.Variable(tf.truncated_normal(shape=[num_features, num_fc_layer1_output], stddev=0.05))
        variable_summaries(fc_weights1)
        fc_biases1 = tf.Variable(tf.constant(0.05, shape=[num_fc_layer1_output]))
        variable_summaries(fc_biases1)

        fc_layer1 = tf.matmul(layer_flat, fc_weights1) + fc_biases1
        tf.summary.histogram('fc_layer1', fc_layer1)

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('fc_bn_1'):
            fc_layer1 = batch_norm(fc_layer1, phase_train)

    # RELU
    with tf.variable_scope('fc_relu_1'):
        fc_layer1 = tf.nn.relu(fc_layer1)

    # DROPOUT LAYER 1
    with tf.variable_scope('dropout_1'):
        fc_layer1_dropout = tf.nn.dropout(fc_layer1, fc_layer1_keep_prob)

    # FULLY CONNECTED LAYER 2
    with tf.variable_scope('fc_2'):
        fc_weights2 = tf.Variable(tf.truncated_normal(shape=[num_fc_layer1_output, num_fc_layer2_output], stddev=0.05))
        variable_summaries(fc_weights2)

        fc_biases2 = tf.Variable(tf.constant(0.05, shape=[num_fc_layer2_output]))
        variable_summaries(fc_biases2)

        fc_layer2 = tf.matmul(fc_layer1_dropout, fc_weights2) + fc_biases2
        tf.summary.histogram('fc_layer2', fc_layer2)

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('fc_bn_2'):
            fc_layer2 = batch_norm(fc_layer2, phase_train)

    # SOFTMAX
    with tf.variable_scope('softmax'):
        y_pred = tf.nn.softmax(fc_layer2)
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        tf.summary.histogram('y_pred', y_pred)

    # COST FUNCTION
    with tf.variable_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer2, labels=y_true))
        # cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
        tf.summary.histogram('cost', cost)

    # GRADIENT DESCENT METHOD - ADAM OPTIMIZER
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # PERFORMANCE MEASURES
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram('accuracy', accuracy)

    return train_step, cost, accuracy, y_pred, y_pred_cls, y_true_cls, placeholders


def train_network(data, train_step, cost, accuracy, num_iterations, train_batch_size, dropout, placeholders, saver, save_dir, plot_dir, log_dir, display_step, figname_suffix):
    with tf.Session() as session:
        print("Session opened!")

        dir = log_dir + "/" + figname_suffix + "/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(dir, session.graph)
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
                               placeholders['phase_train']: True,
                               placeholders['fc_layer1_keep_prob']: dropout
                               }

            # train_step.run(feed_dict_train)
            summary, _ = session.run([merged, train_step], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)

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

        train_writer.close()

        return training_acc, training_cost, testing_acc


def test_network(test_batch_size, placeholders, dropout, saver, save_dir, accuracy,  y_pred_tensor, y_pred_cls, y_true_cls, data):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver.restore(sess=session, save_path=save_dir)

        num_test = len(data.test.images)
        num_iterations = int(num_test / test_batch_size) + 1

        i = 0
        avg_acc = 0
        y_pred_probs = []
        y_pred = []
        y_true = []
        x_all = []
        y_all = []
        while i < num_iterations:
            x_batch, y_batch = data.test.next_batch(test_batch_size)
            x_all += [x_batch]
            y_all += [y_batch]
            feed_dict_test = {placeholders['x']: x_batch,
                              placeholders['y_true']: y_batch,
                              placeholders['fc_layer1_keep_prob']: dropout,
                              placeholders['phase_train']: False}

            y_pred_value, batch_y_pred, batch_y_true, batch_acc = \
                session.run([y_pred_tensor, y_pred_cls, y_true_cls, accuracy], feed_dict=feed_dict_test)
            y_true += batch_y_true.T.tolist()
            y_pred += batch_y_pred.T.tolist()
            y_pred_probs += y_pred_value.T.tolist()
            avg_acc += batch_acc

            i += 1

        acc = float(avg_acc) / num_iterations

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%}"
        print(msg.format(acc))
        x_all = list(itertools.chain.from_iterable(x_all))
        y_all = list(itertools.chain.from_iterable(y_all))

        return y_true, y_pred, acc, y_pred_probs, x_all, y_all


def run_experiment(params, dropout, bn):
    tf.reset_default_graph()

    figname_suffix = "a_"

    dropout_value = 1.0
    if dropout:
        dropout_value = params['dropout']
        figname_suffix += "dropout"
    if bn:
        figname_suffix += "bn"

    train_step, \
    cost, \
    accuracy, \
    y_pred, \
    y_pred_cls, \
    y_true_cls, \
    placeholders = create_network(
                                params['img_size'],
                                params['num_channels'],
                                params['num_classes'],
                                params['shape1'],
                                params['shape2'],
                                params['num_fc_layer1_output'],
                                params['num_fc_layer2_output'],
                                params['learning_rate'],
                                bn
                                )

    saver = tf.train.Saver()
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    (train_acc, cost, test_acc) = train_network(params['data'],
                                              train_step,
                                              cost,
                                              accuracy,
                                              params['num_iterations'],
                                              params['train_batch_size'],
                                              dropout_value,
                                              placeholders,
                                              saver,
                                              params['save_dir'],
                                              params['plot_dir'],
                                              params['log_dir'],
                                              params['display_step'],
                                              figname_suffix)

    cls_true, cls_pred, acc, y_pred_probs, x_all, y_all\
        = test_network(params['test_batch_size'],
                       placeholders,
                       1.0,
                       saver,
                       params['save_dir'],
                       accuracy,
                       y_pred,
                       y_pred_cls,
                       y_true_cls,
                       params['data'])

    plot_confusion_matrix(cls_true=cls_true,
                          cls_pred=cls_pred,
                          output_dir=plot_dir,
                          fig_name="confusion_matrix" + "_" + figname_suffix)

    plot_images(output_dir=plot_dir,
                fig_name="images" + "_" + figname_suffix,
                img_shape=(params['img_size'], params['img_size'], params['num_channels']),
                images=x_all,
                cls_true=y_all,
                cls_pred=cls_pred,
                prob_pred=y_pred_probs,
                logits=None,
                y_pred=None)

    return train_acc, cost, test_acc, figname_suffix, acc


if __name__ == "__main__":
    datasets = [('mnist', 28, 1, 200, 30, 1e-2, 50), ('cifar10', 32, 3, 500, 100, 8e-3, 100)]
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

        save_dir = dataset_dir + "/checkpoints_3_4_" + str(learning_rate) + "_" + str(num_iterations) + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plot_dir = dataset_dir + "/plots_3_4_" + str(learning_rate) + "_" + str(num_iterations) + "/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        log_dir = dataset_dir + "/logs_3_4_" + str(learning_rate) + "_" + str(num_iterations) + "/"
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

        train_batch_size = 1000
        test_batch_size = 1000
        dropout = 0.5

        real_label = 7
        chosen_label = 2

        data = load_data(data_dir, dataset)
        print(data.test.images[0].shape)
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
            'dropout': dropout,
            'display_step': display_step,
            'real_label': real_label,
            'chosen_label': chosen_label
        }

        run_experiment(params, True, True)
