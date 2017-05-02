# encoding: utf-8

"""
@author: ouwj
@position: ouwj-win10
@file: load_data.py
@time: 2017/4/26 14:33
"""

import numpy as np
import sys
from PIL import Image


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
    if dataset == 'data/MNIST':
        from tensorflow.examples.tutorials.mnist import input_data
        return input_data.read_data_sets('MNIST/')
    elif dataset == 'CIFAR':
        dirname = 'data/CIFAR/cifar-10-batches-py/'
        # print(unpickle(dirname+'test_batch'))
        dict = unpickle(dirname+'test_batch')

        # load all data
        # data = dict[b'data'] / 255.0
        # for i in range(1, 6):
        #     dict = unpickle(dirname + 'data_batch_' + str(i))
        #     data = np.vstack((data, dict[b'data'] / 255.0))
        # return data

        # load one class data
        labels = np.array(dict[b'labels'])
        data = dict[b'data'][labels==1] / 255.0
        for i in range(1, 6):
            dict = unpickle(dirname+'data_batch_'+str(i))
            labels = np.array(dict[b'labels'])
            data = np.vstack((data, dict[b'data'][labels==1] / 255.0))
        return data
    elif dataset == 'CELEBA':
        filename = 'data/CELEBA/data_0'
        data = unpickle(filename) / 255.0
        for i in range(1, 6):
            filename = 'data/CELEBA/data_'+str(i)
            data = np.vstack((data, unpickle(filename) / 255.0))
        return data

def check_data(dataset):
    assert(dataset in ['MNIST', 'CIFAR', 'CELEBA'])

    data = load_data('CELEBA')
    n_img = 36
    np_imgs = data[0:n_img]
    np_imgs *= 255
    sep = 3
    np_imgs = np_imgs.astype(np.uint8)

    if dataset == 'CELEBA' or dataset == 'MNIST':
        H = W = int(np_imgs.shape[1] ** 0.5)
        num = int(n_img ** 0.5)
        syn_img = np.zeros((num * H + (num - 1) * sep, num * W + (num - 1) * sep)) * 255
        syn_img = syn_img.astype(np.uint8)
        for i in range(num):
            for j in range(num):
                syn_img[i * (H + sep):(i + 1) * H + i * sep, j * (W + sep):(j + 1) * W + j * sep] = \
                    np_imgs[i * num + j].reshape((H, W))
    else:
        D = 3
        HW = np_imgs.shape[1] / D
        H = W = int(HW ** 0.5)
        num = int(n_img ** 0.5)
        sep = 3
        syn_img = np.ones((num * H + (num - 1) * sep, num * W + (num - 1) * sep, D)) * 255
        syn_img = syn_img.astype(np.uint8)

        for i in range(num):
            for j in range(num):
                syn_img[i * (H + sep):(i + 1) * H + i * sep, j * (W + sep):(j + 1) * W + j * sep, 0:D] = \
                    np_imgs[i * num + j].reshape((D, H, W)).transpose((1, 2, 0))

    im = Image.fromarray(syn_img)
    im.show()

if __name__ == '__main__':
    check_data('CELEBA')


