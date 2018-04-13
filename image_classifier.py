import tensorflow as tf
import numpy as np
import pandas
import keras


# urls to datasets
cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# placeholders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def build_graph():
    """
    Builds neural network model
    :param x: image inputs
    """

    # weights & biases
    w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=1e-1), name='w1')
    b1 = tf.Variable(tf.zeros(64), name='b1')
    w2 = tf.Variable(tf.truncated_normal([5, 5, 3, 128], stddev=1e-1), name='w2')
    b2 = tf.Variable(tf.zeros(64), name='b2')

    # conv layer 1
    conv_layer = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='same')
    bias_add = tf.nn.bias_add(conv_layer, b1)
    activation = tf.nn.leaky_relu(bias_add)

    # pool 1
    pool = tf.nn.max_pool(activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='same')

    # batch norm 1
    batch_norm = tf.nn.batch_normalization(pool)

    # conv layer 2
    conv_layer = tf.nn.conv2d(batch_norm, w2, strides=[1, 1, 1, 1], padding='same')
    bias_add = tf.nn.bias_add(conv_layer, b2)
    activation = tf.nn.leaky_relu(bias_add)

    # pool 2
    pool = tf.nn.max_pool(activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='same')

    # batch norm 2
    batch_norm = tf.nn.batch_normalization(pool)

    # flatten & fc Layers
    flat = tf.layers.flatten(batch_norm)
    fc1 = tf.layers.dense(flat, 300)
    fc2 = tf.layers.dense(fc1, 150)
    output = tf.layers.dense(fc2, 10)

    # TO DO: add dropout, adjust units


def train_model():
    # TO DO


def test_model():
    # TO DO
