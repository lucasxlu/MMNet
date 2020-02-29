import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from config.cfg import cfg


def mkdirs_if_not_exist(dir_name):
    """
    create new folder if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


def numpy_to_img_file():
    """
    convert numpy arrays to image files
    :return:
    """
    fer2013_csv = os.path.join(cfg['root'], 'FER2013', 'fer2013.csv')
    df = pd.read_csv(fer2013_csv)

    for i in range(len(df['Usage'])):
        if df['Usage'][i] == 'Training':
            dir_ = os.path.join(cfg['root'], 'FER2013', 'train', str(df['emotion'][i]))
            mkdirs_if_not_exist(dir_)
            img = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)

            cv2.imwrite(os.path.join(dir_, '%d.jpg' % i), img)
            print('write image %s successfully...' % (os.path.join(dir_, '%d.jpg' % i)))
        elif df['Usage'][i] == 'PrivateTest':
            dir_ = os.path.join(cfg['root'], 'FER2013', 'test', str(df['emotion'][i]))
            mkdirs_if_not_exist(dir_)
            img = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)

            cv2.imwrite(os.path.join(dir_, '%d.jpg' % i), img)
            print('write image %s successfully...' % (os.path.join(dir_, '%d.jpg' % i)))


if __name__ == '__main__':
    numpy_to_img_file()
