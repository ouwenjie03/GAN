# encoding: utf-8

"""
@author: ouwj
@position: ouwj-win10
@file: save_face.py
@time: 2017/5/2 14:45
"""

import pickle
import os
from PIL import Image
import numpy as np
import sys

# face_dir = 'd:/img_align_celeba'
face_dir = '/home/yuedong/Documents/img_align_celeba'
pickle_file = 'data/CELEBA/data_quantize'


def main(part):
    faces = sorted(os.listdir(face_dir))
    start = len(faces) / 6 * part
    end = len(faces) / 6 * (part+1)

    all_data = []
    for f in faces[start:end]:
        print(f)
        im = Image.open(os.path.join(face_dir, f)).convert('L')
        im = im.resize((48, 48))
        # quantization
        im.quantize(colors=8).convert('L')
        arr = np.array(im)
        all_data.append(arr2.reshape((-1)))

    with open(pickle_file+'_'+str(part), 'wb') as fo:
        pickle.dump(np.array(all_data), fo, True)


if __name__ == "__main__":
    part = int(sys.argv[1])
    main(part)
