# encoding: utf-8

"""
@author: ouwj, donald
@position: ouwj-win10
@file: main.py
@time: 2017/4/20 14:16
"""

from model import *
import argparse
import tensorflow as tf
import sys


conf = None


def main(_):
    gan = Gan(conf)
    gan.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR", "CELEBA"], \
                        default="MNIST", \
                        help="input dataset name")
    parser.add_argument("--use_gpu", type=int, choices=[0, 1], \
                        default=0, \
                        help="whether to use gpu or not")
    conf, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
