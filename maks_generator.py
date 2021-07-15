import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, feature
from skimage.color import rgba2rgb,rgb2gray
from skimage.filters import threshold_otsu
from matplotlib import image

#puvodni obrazek
obrazek = image.imread('C:\\Users\\David\\PycharmProjects\\green_hack\\odpadky\\odpadek.jpg')
plt.figure(figsize=(10,5))
plt.imshow(obrazek)

plt.show()

def histogram(im):
  hist = np.zeros(256, dtype=int)
  for i in im.ravel():
    hist[i] += 1
  return hist

h = histogram(obrazek)
# Show histogram
plt.figure(figsize=(5,5))
plt.plot(h)
plt.show()

#urceni prahu
img = obrazek
obrazek = np.mean(obrazek, axis=2)
prah = threshold_otsu(obrazek)+25

#histogram s prahem
plt.figure(figsize=(5,5))
plt.plot(h)
plt.axvline(prah, color = 'r')
plt.show()

#vybereme pouze


binary = obrazek > prah

#binarni obrazek
plt.figure(figsize=(10,5))
plt.imshow(binary, cmap='gray')
plt.show()


plt.imsave("mask.jpg",binary)