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
    gan = Gan(conf.dataset)
    gan.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR"], \
                        default="MNIST", \
                        help="input dataset name")
    conf, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
