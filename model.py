# encoding: utf-8

"""
@author: ouwj, donald
@position: ouwj-win10
@file: model.py
@time: 2017/4/19 17:07
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
from utils import *
from my_config import *
from load_data import *

class Gan:
    def __init__(self, dataset='MNIST'):
        self.dataset = dataset

        # load model's configure
        self.load_config(get_config(dataset))

        # load data
        self.true_data = load_data(self.dataset)

        # setup sample and check point path
        self.sample_dir = os.path.join('samples', dataset)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        self.ckpt_path = os.path.join('check_point', dataset)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.init_param()
        self.build_gan()

        self.n_img = 36
        self.test_random_input = self.get_random_input([self.n_img, self.input_size])

        self.sess = tf.Session()

    def load_config(self, config):
        self.learning_rate = config.learning_rate
        self.momentum = config.momentum
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_train_discriminator = config.n_train_discriminator
        self.n_train_generator = config.n_train_generator

        self.img_size = config.image_size
        self.input_size = config.input_size

        self.g_n_hidden1 = config.g_n_hidden1
        self.g_n_hidden2 = config.g_n_hidden2
        self.d_n_hidden1 = config.d_n_hidden1
        self.d_n_hidden2 = config.d_n_hidden2

    def binarize(self, img):
        # 随机噪声作为对抗样本
        # return (np.random.uniform(size=img.shape) < img).astype(np.float32)
        # return (img > np.array([0.5]*self.img_size)).astype(np.float32)
        return img

    def get_random_input(self, size):
        return np.random.uniform(low=-1, high=1, size=size)

    def init_variable(self, shape, name=None):
        return tf.Variable(tf.random_uniform(shape, minval=-0.1, maxval=0.1), name=name)

    def init_param(self):
        g_n_input = self.input_size
        d_n_input = self.img_size

        self.g_input = tf.placeholder(tf.float32, [None, g_n_input], name='g_input')
        self.d_input = tf.placeholder(tf.float32, [None, d_n_input], name='d_input')

        with tf.variable_scope('generator'):
            g_n_output = self.img_size
            self.g_weight_layer1 = self.init_variable([g_n_input, self.g_n_hidden1], name='layer1')
            self.g_bias_layer1 = self.init_variable([self.g_n_hidden1], name='layer1')
            self.g_weight_layer2 = self.init_variable([self.g_n_hidden1, self.g_n_hidden2], name='layer2')
            self.g_bias_layer2 = self.init_variable([self.g_n_hidden2], name='layer2')
            self.g_weight_layer3 = self.init_variable([self.g_n_hidden2, g_n_output], name='layer3')
            self.g_bias_layer3 = self.init_variable([g_n_output], name='layer3')

        with tf.variable_scope('discriminator'):
            d_n_output = 1
            self.d_weight_layer1 = self.init_variable([d_n_input, self.d_n_hidden1], name='layer1')
            self.d_bias_layer1 = self.init_variable([self.d_n_hidden1], name='layer1')
            self.d_weight_layer2 = self.init_variable([self.d_n_hidden1, self.d_n_hidden2], name='layer2')
            self.d_bias_layer2 = self.init_variable([self.d_n_hidden2], name='layer2')
            self.d_weight_layer3 = self.init_variable([self.d_n_hidden2, d_n_output], name='layer3')
            self.d_bias_layer3 = self.init_variable([d_n_output], name='layer3')

    def generator(self, input):
        with tf.variable_scope('generator'):
            g_hidden1 = tf.add(tf.matmul(input, self.g_weight_layer1), self.g_bias_layer1)
            g_hidden1 = tf.nn.relu(g_hidden1)
            g_hidden2 = tf.add(tf.matmul(g_hidden1, self.g_weight_layer2), self.g_bias_layer2)
            g_hidden2 = tf.nn.relu(g_hidden2)
            g_output = tf.add(tf.matmul(g_hidden2, self.g_weight_layer3), self.g_bias_layer3)
            g_output = tf.nn.sigmoid(g_output)
            return g_output

    def discriminator(self, input):
        with tf.variable_scope('discriminator'):
            d_hidden1 = tf.add(tf.matmul(input, self.d_weight_layer1), self.d_bias_layer1)
            d_hidden1 = tf.nn.dropout(tf.nn.relu(d_hidden1), 0.5)
            d_hidden2 = tf.add(tf.matmul(d_hidden1, self.d_weight_layer2), self.d_bias_layer2)
            d_hidden2 = tf.nn.dropout(tf.nn.relu(d_hidden2), 0.5)
            d_output = tf.add(tf.matmul(d_hidden2, self.d_weight_layer3), self.d_bias_layer3)
            d_output = tf.nn.sigmoid(d_output)
            return d_output

    def build_gan(self):
        self.g_output = self.generator(self.g_input)  # g(z)
        self.d_output_real = self.discriminator(self.d_input)  # d(x)
        self.d_output_gene = self.discriminator(self.g_output)  # d(g(z))

    def g_loss(self):
        return -tf.reduce_mean(tf.log(self.d_output_gene))

    def d_loss(self):
        return -tf.reduce_mean(tf.log(self.d_output_real) + tf.log(1-self.d_output_gene))

    def train(self, is_load=False):
        g_loss = self.g_loss()
        d_loss = self.d_loss()

        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

        t_vars = tf.trainable_variables()
        g_var_list = [var for var in t_vars if 'generator' in var.name]
        g_op = optimizer.minimize(g_loss, var_list=g_var_list)

        d_var_list = [var for var in t_vars if 'discriminator' in var.name]
        d_op = optimizer.minimize(d_loss, var_list=d_var_list)

        saver = tf.train.Saver()

        if is_load:
            self.load_model()
        else:
            self.sess.run(tf.global_variables_initializer())

        d_l = 1
        g_l = 1
        start_idx = 0
        for i in range(self.n_epoch):
            for j in range(self.n_train_discriminator):
                if self.dataset == 'MNIST':
                    true_batch = self.true_data.train.next_batch(self.batch_size)[0] # [0]:images, [1]:labels
                elif self.dataset == 'CIFAR':
                    if start_idx >= len(self.dataset):
                        start_idx -= len(self.dataset)
                    true_batch = self.true_data[start_idx:(start_idx+self.batch_size)]
                else:
                    true_batch = None

                random_batch = self.get_random_input([self.batch_size, self.input_size])
                feed_dict = {self.g_input: random_batch,
                             self.d_input: true_batch}
                _, d_l = self.sess.run([d_op, d_loss], feed_dict)

            for j in range(self.n_train_generator):
                random_batch = self.get_random_input([self.batch_size, self.input_size])
                feed_dict = {self.g_input: random_batch}
                _, g_l = self.sess.run([g_op, g_loss], feed_dict)

            if i % 10 == 0:
                print("%d | g_loss: %f | d_loss: %f" % (i, g_l, d_l))
            if i % 2000 == 0:
                saver.save(self.sess, os.path.join(self.ckpt_path, 'gan.ckpt'), \
                           global_step=int(i/200))
                print("check point saving...")

            if i % 2000 == 0:
                self.sample_images(i)
                print("Iteration: %d, successfully saved samples_MNIST" % i)

    def load_model(self):
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(self.ckpt_path)
        saver.restore(self.sess, ckpt)

    def sample_images(self, cur_epoch):
        self.load_model()
        feed_dict = {self.g_input: self.test_random_input}
        generate_imgs = self.sess.run(self.g_output, feed_dict=feed_dict)
        save_samples(self.dataset, generate_imgs, self.sample_dir, cur_epoch, self.n_img)
