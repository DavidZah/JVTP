import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, feature
from skimage.color import rgba2rgb, rgb2gray
from skimage.filters import threshold_otsu
from matplotlib import image
import os

treshold = 10

path_train = "test_frames/VID_20210715_213435.mp4"
path_test = "test_frames/VID_20210715_213435.mp4"

path_anotation_test = "anotation/val/"
path_anotation_train = "anotation/train/"

path_train_dir = 'images/train/'
path_test_dir = 'images/val/'

def gen_binary(img):
    img = np.mean(img, axis=2)
    prah = threshold_otsu(img)

    binary = img > prah + treshold
    return binary

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    ensure_dir(path_train_dir)
    ensure_dir(path_test_dir)
    ensure_dir(path_anotation_train)
    ensure_dir(path_train_dir)

    for f in os.listdir(path_train_dir):
        os.remove(os.path.join(path_train_dir, f))
    for f in os.listdir(path_test_dir):
        os.remove(os.path.join(path_test_dir, f))

    counter = 0
    with os.scandir(path_train) as entries:
        for entry in entries:
            img = image.imread(entry.path)
            bin_img = gen_binary(img)

            image.imsave(path_anotation_train + str(counter) + '.jpg', bin_img)
            image.imsave(path_train_dir + str(counter) + '.jpg', img)

            counter += 1
    counter = 0
    with os.scandir(path_test) as entries:
        for entry in entries:
            img = image.imread(entry.path)
            bin_img = gen_binary(img)

            image.imsave(path_anotation_test  + str(counter) + '.jpg', bin_img)
            image.imsave(path_test_dir + str(counter) + '.jpg', img)

            counter += 1
    print("done")
