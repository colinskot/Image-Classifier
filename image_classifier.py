import tensorflow as tf
import numpy as np
import data_helper as data


save_model_path = './saved-model'


def build_model():
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
    activation = tf.nn.relu(bias_add)

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
    activation = tf.nn.relu(bias_add)

    # pool 2
    pool = tf.nn.max_pool(activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # dropout 2
    dropout = tf.nn.dropout(pool, keep_prob=0.6)

    # batch norm 2
    # batch_norm = batch_normalization(dropout)

    # flatten & fc Layers
    flat = tf.contrib.layers.flatten(dropout)
    fc1 = tf.layers.dense(flat, 300)
    fc2 = tf.layers.dense(fc1, 150)
    output = tf.layers.dense(fc2, 10)

    return output


def train_model(logits):

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


    print('Training...')
    with tf.Session() as sess:

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # training cycle
        for epoch in range(epochs):
            # Loop over all batches
            for batch_i in range(1, 6):
                for feature_batch, label_batch in data.load_training_batch(batch_i, batch_size):
                    sess.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})

                print('Epoch {}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')

                # calculate loss & validation
                loss = sess.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
                valid_acc = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})

                # print loss & validation
                print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

        # save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)


def test_model():
    # TO DO
    print('Testing Model...')


def batch_normalization(x):
    """
    Normalizes the current batch
    :param x: inputs
    """

    # calculate mean, variance, scale, etc.
    mean, var = tf.nn.moments(x, [0])
    beta = tf.Variable(tf.zeros([x.shape[1]]))
    scale = tf.Variable(tf.ones([x.shape[1]]))
    epsilon = 1e-3

    bn = tf.nn.batch_normalization(x, mean, var, beta, scale, epsilon)

    return bn


if __name__ == '__main__':

    # choose type of data set, download, preprocess
    data.download('cifar10')
    data.preprocess_and_save()

    # hyperparameters
    epochs = 25
    batch_size = 256

    # remove previous weights, bias, inputs, etc.
    tf.reset_default_graph()

    # placeholders
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x')
    y = tf.placeholder(tf.int32, shape=(None, 10), name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # build model
    logits = build_model()

    # train model
    train_model(logits)

    # test model
    test_model()
