# encoding: utf-8

"""
@author: donald, ouwj
@position: donald-elementary
@file: utils.py
@time: 2017/4/25 11:16
"""

from PIL import Image
from os.path import join
import numpy as np


def save_samples(dataset, np_imgs, img_path, cur_epoch, n_img):
    if dataset == 'MNIST' or dataset == 'CELEBA':
        save_samples_gray(np_imgs, img_path, cur_epoch, n_img)
    elif dataset == 'CIFAR':
        save_samples_rgb(np_imgs, img_path, cur_epoch, n_img)

def save_samples_gray(np_imgs, img_path, cur_epoch, n_img):
    """
    synthesize images and save it to img_path
    Args:
        np_imgs: [N, H, W, 3] [0, 1.0] float32
        img_path: str
        cur_epoch: int
        n_imgs: int
    """
    np_imgs *= 255
    np_imgs = np_imgs.astype(np.uint8)
    # quantize to 8 colors
    level = 8
    diff = 256 / level
    scale = 255 / (level - 1)
    np_imgs = scale * (np_imgs / diff)
    # syn image
    H = W = int(np_imgs.shape[1] ** 0.5)
    num = int(n_img ** 0.5)
    sep = 3
    syn_img = np.zeros((num * H + (num - 1) * sep, num * W + (num - 1) * sep)) * 255
    syn_img = syn_img.astype(np.uint8)
    for i in range(num):
        for j in range(num):
            syn_img[i*(H+sep):(i+1)*H+i*sep, j*(W+sep):(j+1)*W+j*sep] = \
                    np_imgs[i*num + j].reshape((H, W))

    im = Image.fromarray(syn_img)
    im.save(join(img_path, "sample_img_%d.jpg" % cur_epoch))



def save_samples_rgb(np_imgs, img_path, cur_epoch, n_img):
    """
    synthesize images and save it to img_path
    Args:
        np_imgs: [N, H, W, 3] [0, 1.0] float32
        img_path: str
        cur_epoch: int
        n_imgs: int
    """
    np_imgs *= 255
    np_imgs = np_imgs.astype(np.uint8)
    D = 3
    HW = np_imgs.shape[1] / D
    H = W = int(HW ** 0.5)
    num = int(n_img ** 0.5)
    sep = 3
    syn_img = np.ones((num * H + (num - 1) * sep, num * W + (num - 1) * sep, D)) * 255
    syn_img = syn_img.astype(np.uint8)
    # with open("image"+str(cur_epoch)+".npy", "wb") as f:
    #     np.save(f, np_imgs[1])

    for i in range(num):
        for j in range(num):
            syn_img[i*(H+sep):(i+1)*H+i*sep, j*(W+sep):(j+1)*W+j*sep, 0:D] = \
                    np_imgs[i*num + j].reshape((D, H, W)).transpose((1, 2, 0))

    im = Image.fromarray(syn_img)
    im.save(join(img_path, "sample_img_%d.jpg" % cur_epoch))
