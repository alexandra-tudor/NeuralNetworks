import numpy as np
import tensorflow as tf
# import cv2
import os
import cifar10
from cifar10 import img_size, num_channels, num_classes
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()

    return temp_batch


def batch_creator(batch_size, dataset_length, dataset_name, input_num_units):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)

    batch_y = None
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y


def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


def sigmaprime1(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


def tanh(x):
    return tf.div(tf.subtract(tf.exp(x, tf.exp(tf.negative(x)))), tf.add(tf.exp(x, tf.exp(tf.negative(x)))))


# def tanhprime(x):
#     return tf.subtract(tf.constant(1.0), tf.multiply(tanh(x), tanh(x)))


def random_batch(images_train, labels_train, train_batch_size):
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch


def plot_confusion_matrix(cls_true, cls_pred, output_dir, fig_name):
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig(output_dir + fig_name + ".png")
    plt.clf()


def plot_images_err(output_dir, fig_name, img_shape, images, cls_true, cls_pred=None, logits=None, y_pred=None):
    errors = list(filter(lambda x: x[0] != x[1], zip(cls_true, cls_pred, images)))
    fig, axes = plt.subplots(6, 6)
    fig.subplots_adjust(hspace=0.6, wspace=1.5)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(errors[i][2].reshape(img_shape), cmap='binary')

        if logits is not None:
            print (logits[i])

        if y_pred is not None:
            print (y_pred[i])

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(errors[i][0])
        else:
            xlabel = "True: {0}, Pred: {1}".format(errors[i][0], errors[i][1])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(output_dir + fig_name+".png")
    plt.clf()


def plot_images(output_dir, fig_name, img_shape, images, cls_true, cls_pred=None, prob_pred=None, logits=None, y_pred=None):
    if img_shape[2] == 1:
        img_shape = img_shape[:-1]

    cls_true = list(map(lambda x: np.argmax(x), cls_true))
    images = list(filter(lambda x: x[0] != x[1], zip(cls_true, cls_pred, images)))
    fig, axes = plt.subplots(5, 5)
    fig.subplots_adjust(hspace=1.6, wspace=3.0)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i][2].reshape(img_shape), cmap='binary')

        if logits is not None:
            print (logits[i])

        if y_pred is not None:
            print (y_pred[i])

        # if prob_pred is not None:
        #     print ("{0}".format(prob_pred[i] * 100))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(images[i][0])
        else:
            xlabel = "True: {0}, Pred: {1}".format(images[i][0], images[i][1])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(output_dir + fig_name+".png")
    plt.clf()


def make_data(cifar10):

    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()

    print(images_train.shape)
    input_nodes = images_train.shape[1]*images_train.shape[2]*images_train.shape[3]

    images_train = images_train.reshape([len(images_train), input_nodes])
    images_test = images_test.reshape([len(images_test), input_nodes])

    class data():
        def __init__(self):
            pass

        class test():
            def __init__(self):
                self.images = images_test
                self.labels = labels_test
                self.cls = cls_test

            def next_batch(self, batch_size, shuffle=False):
                # Number of images (transfer-values) in the training-set.
                num_images = len(labels_test)

                # Create a random index.
                idx = np.random.choice(num_images,
                                       size=batch_size,
                                       replace=False)

                # Use the random index to select random x and y-values.
                # We use the transfer-values instead of images as x-values.
                x_batch = []
                y_batch = []
                for i in idx:
                    x_batch += [images_test[i]]
                    y_batch += [labels_test[i]]

                return x_batch, y_batch

        class train():
            def __init__(self):
                pass

            def next_batch(self, batch_size, shuffle=False):
                # Number of images (transfer-values) in the training-set.
                num_images = len(labels_train)

                # Create a random index.
                idx = np.random.choice(num_images,
                                       size=batch_size,
                                       replace=False)

                # Use the random index to select random x and y-values.
                # We use the transfer-values instead of images as x-values.
                x_batch = []
                y_batch = []
                for i in idx:
                    x_batch += [images_train[i]]
                    y_batch += [labels_train[i]]

                return x_batch, y_batch

    data = data()
    data.test = data.test()
    data.train = data.train()

    return data


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def load_data(data_dir, dataset='mnist'):
    if dataset == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets(data_dir, one_hot=True)

        data.test.cls = np.argmax(data.test.labels, axis=1)
    else:
        cifar10.data_path = data_dir
        cifar10.maybe_download_and_extract()

        data = make_data(cifar10)

    return data


def print_results(cls_pred, cls_true, fig_name_suffix, plot_dir):
    plot_confusion_matrix(cls_true=cls_true,
                          cls_pred=cls_pred,
                          output_dir=plot_dir,
                          fig_name="confusion_matrix" + "_" + fig_name_suffix)


def read_images(image_dir, gray=False):
    images = []
    img_shape = None
    for filename in os.listdir(image_dir):
        if gray:
            img = np.array(cv2.imread(os.path.join(image_dir, filename), 0))
        else:
            img = np.array(cv2.imread(os.path.join(image_dir, filename)))
        img_shape = img.shape
        flatten_shape = 1
        for d in img.shape:
            flatten_shape *= d
        img = img.flatten().reshape(flatten_shape)
        if img is not None:
            images.append(img)

    return images, img_shape
