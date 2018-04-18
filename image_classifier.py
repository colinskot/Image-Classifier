import tensorflow as tf
import numpy as np
import pandas

# urls to datasets
cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# placeholders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def batch_normalization(x):
    """
    Normalizes the current batch
    :param x: inputs
    """

    # calculate mean, variance, scale, etc.
    mean, var = tf.nn.moments(x,[0])
    beta = tf.Variable(tf.zeros([x.shape[1]]))
    scale = tf.Variable(tf.ones([x.shape[1]]))
    epsilon = 1e-3

    bn = tf.nn.batch_normalization(x, mean, var, beta, scale, epsilon)

    return bn


def build_graph(x):
    """
    Builds neural network model
    :param x: image inputs
    """

    # weights & biases
    w1 = tf.Variable(tf.truncated_normal([5, 5, int(x.shape[3]), 64], stddev=1e-1), name='w1')
    b1 = tf.Variable(tf.ones(64), name='b1')

    # conv layer 1
    conv_layer = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    bias_add = tf.nn.bias_add(conv_layer, b1)
    activation = tf.nn.leaky_relu(bias_add)

    # pool 1
    pool = tf.nn.max_pool(activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # dropout 1
    dropout = tf.nn.dropout(pool, keep_prob=0.8)

    # batch norm 1
    # batch_norm = batch_normalization(dropout)

    # weights & biases
    w2 = tf.Variable(tf.truncated_normal([5, 5, int(dropout.shape[3]), 128], stddev=1e-1), name='w2')
    b2 = tf.Variable(tf.ones(128), name='b2')

    # conv layer 2
    conv_layer = tf.nn.conv2d(dropout, w2, strides=[1, 1, 1, 1], padding='SAME')
    bias_add = tf.nn.bias_add(conv_layer, b2)
    activation = tf.nn.leaky_relu(bias_add)

    # pool 2
    pool = tf.nn.max_pool(activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # dropout 2
    dropout = tf.nn.dropout(pool, keep_prob=0.6)

    # batch norm 2
    # batch_norm = batch_normalization(dropout)

    # flatten & fc Layers
    flat = tf.layers.flatten(dropout)
    fc1 = tf.layers.dense(flat, 300)
    fc2 = tf.layers.dense(fc1, 150)
    output = tf.layers.dense(fc2, 10)

    return output


def train_model():
    # TO DO
    pass


def test_model():
    # TO DO
    pass


build_graph(x)
