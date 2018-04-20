import os
import pickle
import tarfile
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib.request import urlretrieve


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


class Data():

    # cifar10
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar10_gz_path = 'data/cifar10.tar.gz'
    cifar10_extracted_path = 'data/cifar-10-batches-py'
    cifar10_preprocessed_path = 'data/preprocessed/'
    cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, data_name='cifar10'):

        self.data_name = data_name

        # create all required directories
        if not os.path.exists('data'):
            os.makedirs('data')

        if not os.path.exists('data/preprocessed'):
            os.makedirs('data/preprocessed')


    def download_data(self):
        """
        Download & extract data based on data_name attribute
        """
        if self.data_name == 'cifar10':
            if not os.path.isfile(cifar10_gz_path):
                with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
                    urlretrieve(cifar10_url, cifar10_gz_path, pbar.hook)

            if not os.path.isfile(cifar10_extracted_path):
                with tarfile.open(cifar10_gz_path) as tar:
                    tar.extractall('data')
                    tar.close()


    def preprocess_and_save(self):
        """
        Preprcesses the data set and saves the file
        """
        valid_features = []
        valid_labels = []

        for batch_i in range(1, 6):

            # open data file
            with open(cifar10_extracted_path + '/data_batch_' + str(batch_i), mode='rb') as file:
                batch = pickle.load(file, encoding='latin1')

            # shape data
            features = batch['data'].reshape(len(batch['data']), 3, 32, 32).transpose(0, 2, 3, 1)
            labels = batch['labels']

            validation_count = int(len(features) * 0.1) # 10% of features

            # normalize training features & one_hot_encode training labels
            train_features = _normalize(np.array(features[:-validation_count]))
            train_labels = _one_hot(np.array(labels[:-validation_count]))

            # save preprocessed training data
            pickle.dump((train_features, train_labels), open(cifar10_preprocessed_path + 'preprocess_batch_' + str(batch_i) + '.p', mode='wb'))

            # add to validation features & Labels
            valid_features.extend(features[-validation_count:])
            valid_labels.extend(labels[-validation_count:])

        # normalize validation features & one_hot_encode validation labels
        valid_features = _normalize(np.array(valid_features))
        valid_labels = _one_hot(np.array(valid_labels))

        # save preprocessed validation data
        pickle.dump((valid_features, valid_labels), open(cifar10_preprocessed_path + 'preprocess_validation.p', mode='wb'))

        # preprocess & save test data
        with open(cifar10_extracted_path + '/test_batch', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # shape test data
        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']

        # normalize test features & one_hot_encode test labels
        test_features = _normalize(test_features)
        test_labels = _one_hot(test_labels)

        # save test data
        pickle.dump((test_features, test_labels), open(cifar10_preprocessed_path + 'preprocess_test.p', mode='wb'))


    def load_training_batch(batch_id, batch_size):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """
        filename = cifar10_preprocessed_path + 'preprocess_batch_' + str(batch_id) + '.p'
        features, labels = pickle.load(open(filename, mode='rb'))

        # Return the training data in batches of size <batch_size> or less
        return _batch(features, labels, batch_size)


    def _batch(features, labels, batch_size):
        """
        Split features and labels into batches
        """
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]


    def _normalize(x):
        """
        Normalizes the image
        :param x: image
        """
        return x/255

    def _one_hot(x):
        """
        One hot encodes the labels
        :param x: labels
        """
        # return diagonal array rows in position x.reshape(-1)
        arr = np.array(x).reshape(-1)
        return np.eye(10, dtype=int)[arr]


data = Data('cifar10')
data.download_data()
data.preprocess_and_save()
