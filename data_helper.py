import os
import pickle
import tarfile
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib.request import urlretrieve


# download progress bar
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


# cifar10
cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
cifar10_gz_path = 'data/cifar10.tar.gz'
cifar10_extracted_path = 'data/cifar-10-batches-py'
cifar10_preprocessed_path = 'data/preprocessed'
cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def _make_data_dirs():

    # create all required directories
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists('data/preprocessed'):
        os.makedirs('data/preprocessed')


def download(data_name):
    """
    Download & extract data based on data_name
    """
    # first make sure to create directories
    _make_data_dirs()

    print('Data name: ' + data_name)

    # download & extract based on name of data set
    if data_name == 'cifar10':
        if not os.path.isfile(cifar10_gz_path):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
                urlretrieve(cifar10_url, cifar10_gz_path, pbar.hook)

        if not os.path.isfile(cifar10_extracted_path):
            with tarfile.open(cifar10_gz_path) as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, "data")
                tar.close()


def preprocess_and_save():
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


def _normalize(images):
    """
    Normalizes the images
    """
    return images/255

def _one_hot(labels):
    """
    One hot encodes the labels
    """
    arr = np.array(labels).reshape(-1)
    return np.eye(10, dtype=int)[arr]
