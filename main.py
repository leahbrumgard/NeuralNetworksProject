import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D

# TODO: load test file (if you want)

def main():
    xtrain1 = np.load('/scratch/tkyaw1/outfile.npz')
    ytrain1 = np.load('/scratch/tkyaw1/labels.npz')
    neural_net = Sequential()

    # neural_net.add(Conv2D())
    # TODO: how to add conv/dropout layers. neural net fit dimensions??
    # TODO: first input to dense layer? also, how to do the .files is not relying on ['arr_0']
    
    neural_net.add(Dense(512, activation = 'hard_sigmoid', input_shape = (768, 1050, 3)))
    # neural_net.add(Dropout(0.2)) #added
    # neural_net.add(Dense(10, activation='softmax'))
    neural_net.summary()
    # xtrain = xtrain1.files
    # ytrain = ytrain1.files
    xtrainfinal = xtrain1['arr_0']
    ytrainfinal = ytrain1['arr_0']
    # print x
    neural_net.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])


    # history = neural_net.fit(xtrain1, ytrain1, verbose=1, validation_data=(xtrainfinal, ytrainfinal), epochs=10)

    loss, accuracy = neural_net.evaluate(xtrainfinal, ytrainfinal, verbose=0)
    # print "accuracy: {}%".format(accuracy*100)

    keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
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
