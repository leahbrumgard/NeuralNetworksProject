# import cv2
import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.models import Model
from tensorflow.python import *
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D
from keras.layers.core import Flatten
import pickle
from PIL import Image
import json, codecs
from numpy import savetxt
# import tensorflow as tf


#ndimage.imread("filename")
#plt.imshow(a)
def train_label_img(img):
    word_label = img.split('.')[-3] #get dog/cat out of label
    if word_label == "dog":
        return [0,1]
    elif word_label == "cat":
        return [1,0]

def main():
    pictures = ls("/scratch/tkyaw1/train/")

    maxWidth = 0
    maxHeight = 0
    for p in pictures:
        filename = "/scratch/tkyaw1/train/" + p
        im = Image.open(filename)
        width = im.size[0] # im.size returns (width, height) tuple
        height = im.size[1]
        if width>maxWidth:
            maxWidth = width
        if height>maxHeight:
            maxHeight = height

    xtrain = []
    ytrain = []
    i= 0
    for pic in pictures:
        if i==2000:
            break
        # trainPic = []
        trainPic = np.zeros([maxHeight, maxWidth, 3])
        filename = "/scratch/tkyaw1/train/" + pic
        a = ndimage.imread(filename)
        print a.shape
        im = Image.open(filename)
        pWidth = im.size[0]
        pHeight = im.size[1]
        trainPic[0:pHeight,0:pWidth, 0:3] = a
        # plt.imshow(a)
        # plt.show()
        label = train_label_img(filename)
        ytrain.append(label)
        # trainPic.append(a) #what is this???
        xtrain.append(trainPic)
        i+=1
    # xtrain = np.array(xtrain)
    # print a
    # ytrain = np.array(ytrain)
    plt.imshow(a)
    plt.show()


    # with open('outfile', 'w') as fp:
    #     pickle.dump(xtrain)
    # savetxt("output.txt", xtrain)

    # b = xtrain.tolist()
    # print b[0]
    # with open('data.txt', 'w') as outfile:
    #     json.dump(b, outfile)

    # for pic in xtrain:


    # plt.imshow(a)
    # plt.show()

    # pictures = ls("/scratch/tkyaw1/test/")
    # xtest = []
    # for pic in pictures:
    #     testPic = []
    #     filename = "/scratch/tkyaw1/test/" + pic
    #     a = ndimage.imread(filename)
    #     testPic.append(a)
    #     xtest.append(testPic)

    print "data shapes"
    print "  xtrain:", xtrain.shape
    # print "  xtrain:", xtrain.dtype
    # print "  xtest:", np.array(xtest).shape
    print "  ytrain:", ytrain.shape
    # print "  ytest: don't have it."

    # model = Model(inputs=[xtrain], outputs=[ytrain])


    # neural_net.add(Dense(512, activation = 'hard_sigmoid')) #added
    # neural_net.add(Dropout(0.2)) #added
    # neural_net.add(Dense(10, activation='softmax'))
    # neural_net.summary()
    #
    # neural_net.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    # history = neural_net.fit(x_train_images, y_train_vectors, verbose=1, validation_data=(x_test_images, y_test_vectors), epochs=10)
    #
    # loss, accuracy = neural_net.evaluate(x_test_images, y_test_vectors, verbose=0)
    # print "accuracy: {}%".format(accuracy*100)

    # keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
main()

# TRAIN_DIR = '/scratch/tkyaw1/train'
# TEST_DIR = '/scratch/tkyaw1/test'
# IMG_SIZE = 50 #50-by-50
#
#
# def label_img(img):
#     word_label = img.split('.')[-3] #get dog/cat out of label
#     if word_label == "dog":
#         return [0,1]
#     elif word_label == "cat":
#         return [1,0]
# def create_test_data():
#     test_data = []
#     for img in os.listdir(TEST_DIR):
#         path = os.path.join(TEST_DIR, img)
#         img_num = img.split('.')[0]
#         img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#         test_data.append([np.array(img), img_num])
#     shuffle(test_data)
#     np.save('/scratch/tkyaw1/test_data.npy', test_data)
#     return test_data
#
#
# def create_train_data():
#     train_data = []
#     for img in os.listdir(TRAIN_DIR):
#         label = label_img(img)
#         path = os.path.join(TRAIN_DIR, img)
#         img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#         train_data.append([np.array(img), np.array(label)])
#     shuffle(train_data)
#     np.save('/scratch/tkyaw1/train_data.npy', train_data)
#     return train_data
#
# create_test_data()
