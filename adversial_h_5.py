from datetime import timedelta
from copy import deepcopy

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import itertools

from utils import variable_summaries, plot_confusion_matrix, plot_images, load_data, read_images


def test_adversial_images(y_pred, saver, save_dir, x, keep_prob, phase_train, image_list, img_shape):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(sess=session, save_path=save_dir)

        prob = y_pred.eval(feed_dict={x: list(image_list),
                                      keep_prob: 1.0,
                                      phase_train: True
                                      })

        pred_list = np.zeros(len(image_list)).astype(int)
        pct_list = np.zeros(len(image_list)).astype(int)
        result = []
        for i in range(len(prob)):
            pred_list[i] = np.argmax(prob[i])
            pct_list[i] = prob[i][pred_list[i]] * 100

            if len(img_shape) == 3 and img_shape[2] == 1:
                img_shape = img_shape[:-1]
            image_ex2 = image_list[i].reshape(img_shape)

            result += [(pred_list[i], pct_list[i], image_ex2)]

    return result


def plot_predictions(step, fig, rows, cols, x, phase_train, img_shape, y_pred, image_list, keep_prob):

    prob = y_pred.eval(feed_dict={x: image_list,
                                  keep_prob: 1.0,
                                  phase_train: True
                                  })

    pred_list = np.zeros(len(image_list)).astype(int)
    pct_list = np.zeros(len(image_list)).astype(int)
    axes = []
    for i in range(len(prob)):
        pred_list[i] = np.argmax(prob[i])
        pct_list[i] = prob[i][pred_list[i]] * 100

        if len(img_shape) == 3 and img_shape[2] == 1:
            img_shape = img_shape[:-1]
        image_ex2 = image_list[i].reshape(img_shape)
        ax = fig.add_subplot(rows, cols, 3*step + i + 1)
        ax.imshow(image_ex2, cmap='binary')
        ax.set_title('Label: {0} \nProb: {1}%'.format(pred_list[i], pct_list[i]))
        axes += [ax]

    return image_list[2].reshape(img_shape)


def create_plot_adversarial_images(x, keep_prob, phase_train, x_image, img_shape, y_conv, y_label, lr, n_steps, saver, output_dir, save_dir, fig_name):
    x_image = np.reshape(x_image, (1, x_image.shape[0]))
    original_image = deepcopy(x_image)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_conv)
    derivative = tf.gradients(loss, x)

    derivative_non = tf.clip_by_value(derivative, -0.01, 0.01)
    derivative = derivative - derivative_non

    image_adv = tf.stop_gradient(x - tf.sign(derivative) * lr)
    image_adv = tf.clip_by_value(image_adv, 0, 1)  # prevents -ve values creating 'real' image

    fig = plt.figure(figsize=(25, 25))
    plt.clf()

    cols = 6
    rows = 6

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(sess=session, save_path=save_dir)
        x_images = []
        for step in range(n_steps):
            dydx = session.run(derivative, {x: x_image,
                                            keep_prob: 1.0,
                                            phase_train: False
                                            })
            x_adv = session.run(image_adv, {x: x_image,
                                            keep_prob: 1.0,
                                            phase_train: False
                                            })

            x_image = np.reshape(x_adv, (1, original_image.shape[1]))
            img_adv_list = original_image
            img_adv_list = np.append(img_adv_list, dydx[0], axis=0)
            img_adv_list = np.append(img_adv_list, x_image, axis=0)
            x_images += [plot_predictions(step, fig, rows, cols, x, phase_train, img_shape, y_conv, img_adv_list, keep_prob)]

        plt.savefig(output_dir + fig_name + ".png")

        images_dir = output_dir + fig_name
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        i = 0
        for x in x_images:
            plt.imsave(images_dir + "/" + str(i) + ".png", x, cmap='binary')
            i += 1


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


def create_network2(img_size, num_channels, num_classes, shape1, shape2, num_fc_layer1_output, num_fc_layer2_output, learning_rate, bn=False):

    # PLACEHOLDER VARIABLES
    x2 = tf.placeholder(tf.float32, shape=[None, img_size * img_size * num_channels], name='x2')
    x_image2 = tf.reshape(x2, [-1, img_size, img_size, num_channels])

    y_true2 = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true2')
    y_true_cls2 = tf.argmax(y_true2, dimension=1)

    fc_layer1_keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')

    phase_train2 = tf.placeholder(tf.bool, name='phase_train2')

    placeholders2 = {
        'x': x2,
        'x_image': x_image2,
        'y_true': y_true2,
        'y_true_cls': y_true_cls2,
        'fc_layer1_keep_prob': fc_layer1_keep_prob2,
        'phase_train': phase_train2
    }

    # CONVOLUTIONAL LAYER 1
    with tf.variable_scope('conv_12'):
        conv_weights12 = tf.Variable(tf.truncated_normal(shape1, stddev=0.05))
        conv_biases12 = tf.Variable(tf.constant(0.05, shape=[shape1[3]]))

        conv_layer12 = tf.nn.conv2d(input=x_image2, filter=conv_weights12, strides=[1, 1, 1, 1], padding='SAME') + conv_biases12

    # POOLING LAYER 1
    with tf.variable_scope('pool_12'):
        conv_layer12 = tf.nn.max_pool(value=conv_layer12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('bn_12'):
            conv_layer12 = batch_norm(conv_layer12, phase_train2)

    # RELU
    with tf.variable_scope('relu_12'):
        conv_layer12 = tf.nn.relu(conv_layer12)

    # CONVOLUTIONAL LAYER 2
    with tf.variable_scope('conv_22'):
        conv_weights22 = tf.Variable(tf.truncated_normal(shape2, stddev=0.05))
        conv_biases22 = tf.Variable(tf.constant(0.05, shape=[shape2[3]]))

        conv_layer22 = tf.nn.conv2d(input=conv_layer12, filter=conv_weights22, strides=[1, 1, 1, 1], padding='SAME') + conv_biases22

    # POOLING LAYER 2
    with tf.variable_scope('pool_2'):
        conv_layer22 = tf.nn.max_pool(value=conv_layer22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('bn_2'):
            conv_layer22 = batch_norm(conv_layer22, phase_train2)

    # RELU
    with tf.variable_scope('relu_2'):
        conv_layer22 = tf.nn.relu(conv_layer22)

    # FLATTEN LAYER
    with tf.variable_scope('flatten'):
        layer_shape2 = conv_layer22.get_shape()
        num_features2 = layer_shape2[1:4].num_elements()  # [num_images, img_height * img_width * num_channels]

        layer_flat2 = tf.reshape(conv_layer22, [-1, num_features2])

    # FULLY CONNECTED LAYER 1
    with tf.variable_scope('fc_1'):
        fc_weights12 = tf.Variable(tf.truncated_normal(shape=[num_features2, num_fc_layer1_output], stddev=0.05))
        variable_summaries(fc_weights12)
        fc_biases12 = tf.Variable(tf.constant(0.05, shape=[num_fc_layer1_output]))
        variable_summaries(fc_biases12)

        fc_layer12 = tf.matmul(layer_flat2, fc_weights12) + fc_biases12
        tf.summary.histogram('fc_layer12', fc_layer12)

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('fc_bn_12'):
            fc_layer12 = batch_norm(fc_layer12, phase_train2)

    # RELU
    with tf.variable_scope('fc_relu_12'):
        fc_layer12 = tf.nn.relu(fc_layer12)

    # DROPOUT LAYER 1
    with tf.variable_scope('dropout_12'):
        fc_layer1_dropout2 = tf.nn.dropout(fc_layer12, fc_layer1_keep_prob2)

    # FULLY CONNECTED LAYER 2
    with tf.variable_scope('fc_22'):
        fc_weights22 = tf.Variable(tf.truncated_normal(shape=[num_fc_layer1_output, num_fc_layer2_output], stddev=0.05))
        variable_summaries(fc_weights22)

        fc_biases22 = tf.Variable(tf.constant(0.05, shape=[num_fc_layer2_output]))
        variable_summaries(fc_biases22)

        fc_layer22 = tf.matmul(fc_layer1_dropout2, fc_weights22) + fc_biases22
        tf.summary.histogram('fc_layer22', fc_layer22)

    # BATCH NORMALIZATION
    if bn:
        with tf.variable_scope('fc_bn_2'):
            fc_layer22 = batch_norm(fc_layer22, phase_train2)

    # SOFTMAX
    with tf.variable_scope('softmax2'):
        y_pred2 = tf.nn.softmax(fc_layer22)
        y_pred_cls2 = tf.argmax(y_pred2, dimension=1)

        tf.summary.histogram('y_pred2', y_pred2)

    # COST FUNCTION
    with tf.variable_scope('cost2'):
        cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer22, labels=y_true2))
        # cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
        tf.summary.histogram('cost2', cost2)

    # GRADIENT DESCENT METHOD - ADAM OPTIMIZER
    train_step2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost2)

    # PERFORMANCE MEASURES
    correct_prediction2 = tf.equal(y_pred_cls2, y_true_cls2)
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
    tf.summary.histogram('accuracy2', accuracy2)

    return train_step2, cost2, accuracy2, y_pred2, y_pred_cls2, y_true_cls2, placeholders2


def train_network(data, train_step, cost, accuracy, num_iterations, train_batch_size, dropout, placeholders, saver, save_dir, plot_dir, log_dir, display_step, figname_suffix):
    with tf.Session() as session:
        print("Session opened!")

        dir = log_dir + "/" + figname_suffix + "/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(dir, session.graph)
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

        # train_writer.close()

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

        # y_all_label = list(map(lambda x: np.argmax(x), y_all))
        # chosen = list(filter(lambda x: x[0] == params['real_label'], zip(y_all_label, x_all)))
        # chosen_class = np.zeros(10, dtype=np.float32)
        # chosen_class[params['chosen_label']] = 1.0
        # image_index = np.random.randint(len(chosen) - 1)
        # create_plot_adversarial_images(x=placeholders['x'],
        #                                keep_prob=placeholders['fc_layer1_keep_prob'],
        #                                phase_train=placeholders['phase_train'],
        #                                img_shape=(params['img_size'], params['img_size'], params['num_channels']),
        #                                x_image=chosen[image_index][1],
        #                                y_conv=y_pred_cls,
        #                                y_label=chosen_class,
        #                                lr=0.05,
        #                                n_steps=12,
        #                                saver=saver,
        #                                output_dir=params['plot_dir'],
        #                                save_dir=params['save_dir'],
        #                                fig_name=str(chosen_label) + "_" + str(real_label) + "_test",
        #                                )

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

    y_all_label = list(map(lambda x: np.argmax(x), y_all))
    chosen = list(filter(lambda x: x[0] == params['real_label'], zip(y_all_label, x_all)))
    chosen_class = np.zeros(10)
    chosen_class[params['chosen_label']] = 1
    image_index = np.random.randint(len(chosen)-1)
    create_plot_adversarial_images(x=placeholders['x'],
                                   keep_prob=placeholders['fc_layer1_keep_prob'],
                                   phase_train=placeholders['phase_train'],
                                   img_shape=(params['img_size'], params['img_size'], params['num_channels']),
                                   x_image=chosen[image_index][1],
                                   y_conv=y_pred,
                                   y_label=chosen_class,
                                   lr=0.05,
                                   n_steps=12,
                                   saver=saver,
                                   output_dir=params['plot_dir'],
                                   save_dir=params['save_dir'],
                                   fig_name=str(chosen_label) + "_" + str(real_label),
                                   )

    # create a second network, train and test both
    train_step2, \
    cost2, \
    accuracy2, \
    y_pred2, \
    y_pred_cls2, \
    y_true_cls2, \
    placeholders2 = create_network2(
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

    (train_acc2, cost2, test_acc2) = train_network(params['data'],
                                                train_step2,
                                                cost2,
                                                accuracy2,
                                                params['num_iterations'],
                                                params['train_batch_size'],
                                                dropout_value,
                                                placeholders2,
                                                saver,
                                                params['save_dir2'],
                                                params['plot_dir'],
                                                params['log_dir'],
                                                params['display_step'],
                                                figname_suffix)
    if params['dataset'] == 'mnist':
        gray = True
    else:
        gray = False
    image_list, img_shape = read_images(params['plot_dir'] + str(chosen_label) + "_" + str(real_label) + "/", gray)
    result1 = test_adversial_images(y_pred, saver, params['save_dir'], placeholders['x'], placeholders['fc_layer1_keep_prob'], placeholders['phase_train'], image_list, img_shape)
    result2 = test_adversial_images(y_pred2, saver, params['save_dir2'], placeholders2['x'], placeholders2['fc_layer1_keep_prob'], placeholders2['phase_train'], image_list, img_shape)

    fig = plt.figure(figsize=(25, 35))
    plt.clf()
    rows = 6
    cols = 4
    for i in range(1, len(result1)):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(result1[i][2], cmap='binary')
        ax.set_title('Label1: {0} Prob1: {1}%'.format(result1[i][0], result1[i][1]))

        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(result2[i][2], cmap='binary')
        ax.set_title('Label2: {0} Prob2: {1}%'.format(result2[i][0], result2[i][1]))

        i += 2

    plt.savefig(params['plot_dir'] + str(chosen_label) + "_" + str(real_label) + "/compare.png")

    return train_acc, cost, test_acc, figname_suffix, acc


if __name__ == "__main__":
    datasets = [('mnist', 28, 1, 2000, 30, 1e-2, 50), ('cifar10', 32, 3, 5000, 100, 8e-3, 100)]
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

        save_dir2 = dataset_dir + "/checkpoints2_3_4_" + str(learning_rate) + "_" + str(num_iterations) + "/"
        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)

        plot_dir = dataset_dir + "/plots_3_4_" + str(learning_rate) + "_" + str(num_iterations) + "/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_dir2 = dataset_dir + "/plots2_3_4_" + str(learning_rate) + "_" + str(num_iterations) + "/"
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

        real_label = 2
        chosen_label = 4

        data = load_data(data_dir, dataset)
        print(data.test.labels[:5])
        print(data.test.cls[:5])
        params = {
            'dataset': dataset,
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
            'save_dir2': save_dir2,
            'plot_dir': plot_dir,
            'plot_dir2': plot_dir2,
            'log_dir': log_dir,
            'train_batch_size': train_batch_size,
            'test_batch_size': test_batch_size,
            'dropout': dropout,
            'display_step': display_step,
            'real_label': real_label,
            'chosen_label': chosen_label
        }

        run_experiment(params, True, True)
