# encoding: utf-8

"""
@author: ouwj
@position: ouwj-win10
@file: my_config.py
@time: 2017/4/25 14:12
"""


class MNIST_Config:
    # parameters of model
    learning_rate = 0.1
    momentum = 0.5
    batch_size = 64
    n_epoch = 20000
    n_train_discriminator = 1
    n_train_generator = 1

    # parameters of input & output
    image_size = 28*28
    input_size = 10*10

    # hidden parameters of generator
    g_n_hidden1 = 1200
    g_n_hidden2 = 1200

    # hidden parameters of discriminator
    d_n_hidden1 = 240
    d_n_hidden2 = 240


