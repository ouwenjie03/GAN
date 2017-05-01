# encoding: utf-8

"""
@author: ouwj
@position: ouwj-win10
@file: load_data.py
@time: 2017/4/26 14:33
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys


def unpickle(file):
    # PYTHON 3
    if sys.version_info.major == 3:
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    # PYTHON 2
    else:
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict


def load_data(dataset='MNIST'):
    if dataset == 'MNIST':
        return input_data.read_data_sets('MNIST/')
    elif dataset == 'CIFAR':
        dirname = 'CIFAR/cifar-10-batches-py/'
        # print(unpickle(dirname+'test_batch'))
        dict = unpickle(dirname+'test_batch')

        # load all data
        # data = dict[b'data'] / 255.0
        # for i in range(1, 6):
        #     dict = unpickle(dirname + 'data_batch_' + str(i))
        #     data = np.vstack((data, dict[b'data'] / 255.0))
        # return data

        # load one class data
        labels = dict[b'label']
        data = []
        for i in range(len(labels)):
            if labels[i] == 1:
                data.append(dict[b'data'] / 255.0)
        for i in range(1, 6):
            dict = unpickle(dirname+'data_batch_'+str(i))
            labels = dict[b'label']
            for j in range(len(labels)):
                if labels[i] == 1:
                    data.append(dict[b'data'] / 255.0)
            # data = np.vstack((data, dict[b'data'] / 255.0))
        # return data
        return np.array(data)


if __name__ == '__main__':
    data = load_data('CIFAR')
    print(data[0:5, :])
