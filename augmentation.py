import os
import time
import warnings
import glob
import torch

warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp, resize, rescale
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt

SOURCE_DIR = 'C:/Users/User/Desktop/tmp/orig'
SOURCE_IMAGE = '10.jpg'
TARGET_DIR = 'C:/Users/User/Desktop/tmp/result'

X1 = 220
X2 = 870
Y1 = 650
Y2 = 1280


def main():
    # print(len([name for name in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, name))]))
    # print([name for name in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, name))])

    image = io.imread(os.path.join(SOURCE_DIR, SOURCE_IMAGE))
    # print(image.shape)

    # rotated = rotate(image, angle=10, mode='wrap')

    # transform = AffineTransform(translation=(100, 200))
    # wrap_shift = warp(image, transform, mode='wrap')

    # flip_left_right = np.fliplr(image)
    # flip_up_down = np.flipud(image)

    # std_dev = 0.05
    # noisy_random = random_noise(image, var=std_dev ** 2)
    # blurred = gaussian(image, sigma=std_dev, multichannel=True)

    # h = image.shape[0]
    # w = image.shape[0]
    # resized = resize(image, (int(h * 0.3), int(w * 0.3)))

    # rescaled = rescale(image, 0.3, multichannel=True)

    # crop_and_save()


def crop_and_save(source_dir=SOURCE_DIR):
    for name in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, name)):
            image = io.imread(os.path.join(source_dir, name))
            cropped = image[X1:X2, Y1:Y2]
            s(cropped, name=name)


def delete_all_files(dir):  # access denied
    files = glob.glob(os.path.join(dir, '*'))
    for f in files:
        os.remove(f)


def v(image):
    io.imshow(image)
    plt.show()


def s(image, target_dir=TARGET_DIR, name=str(time.time())):
    io.imsave(os.path.join(target_dir, '{}.jpg'.format(name.split('.')[0])), image)


if __name__ == '__main__':
    main()
