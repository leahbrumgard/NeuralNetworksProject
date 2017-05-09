"""
Preprocessing file. Reads in all training images and creates 3 dimensional
arrays from them, putting them in 20 different files each with 1000 image arrays
of dimension (768, 1050, 3). Also creates a label file for all 20000 images.
"""

import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from random import shuffle


#ndimage.imread("filename")
#plt.imshow(a)
def train_label_img(img):
    word_label = img.split('.')[-3] #get dog/cat out of label
    if word_label == "/scratch/tkyaw1/smallSubset/dog":
        return [0,1]
    elif word_label == "/scratch/tkyaw1/smallSubset/cat":
        return [1,0]

def main():
    pictures = ls("/scratch/tkyaw1/smallSubset")
    shuffle(pictures)
    maxWidth = 0
    maxHeight = 0
    for p in pictures:
        filename = "/scratch/tkyaw1/smallSubset/" + p
        im = Image.open(filename)
        width = im.size[0] # im.size returns (width, height) tuple
        height = im.size[1]
        if width>maxWidth:
            maxWidth = width
        if height>maxHeight:
            maxHeight = height

    #TODO: the last 1024
    ytrain = []
    xtrain = []
    start = 0
    end = 1000
    for pic in pictures[start:end]:
        # trainPic = []
        trainPic = np.zeros([maxHeight, maxWidth, 3])
        filename = "/scratch/tkyaw1/smallSubset/" + pic
        a = ndimage.imread(filename)
        im = Image.open(filename)
        pWidth = im.size[0]
        pHeight = im.size[1]
        trainPic[0:pHeight,0:pWidth, :] = a
        # plt.imshow(a)
        # plt.show()
        label = train_label_img(filename)
        ytrain.append(label)
        # trainPic.append(a) #what is this???
        xtrain.append(trainPic)
    np.savez_compressed('/scratch/tkyaw1/smallSubset.npz', xtrain)
    np.savez_compressed('/scratch/tkyaw1/smallLabels.npz', ytrain)

main()
