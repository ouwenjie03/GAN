# encoding: utf-8

"""
@author: ouwj
@position: ouwj-win10
@file: main.py
@time: 2017/4/20 14:16
"""

from model import *


# gan = Gan('MNIST')
gan = Gan('CIFAR')

gan.train()
