import tensorflow as tf
import numpy as np

save_model_path = './image_classification'

# placeholders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')


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


def train_model():

    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                # calculate loss & validation
                loss = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
                valid_acc = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})

                # print loss & validation
                print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)


def test_model():
    # TO DO
    pass


build_graph(x)


if __name__ == '__main__':

    # preprocess training, validation, testing data
    helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

    # Load the Preprocessed Validation data
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
