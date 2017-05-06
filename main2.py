import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from numpy import argmax, zeros, logical_not

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D, Flatten

# TODO: load test file (if you want)
# NOTE: relu for non output layers, sigmoid for output layers ?? not sure y
# NOTE: adding another pair of pooling/conv layer dropped accuracy a lot

def settingItUp():
    neural_net = Sequential()

    neural_net.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (768, 1050, 3)))
    neural_net.add(Conv2D(128, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    neural_net.add(Conv2D(128, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    neural_net.add(Flatten())
    neural_net.add(Dense(32, activation = 'relu'))
    neural_net.add(Dropout(0.5))
    neural_net.add(Dense(2, activation = 'sigmoid'))
    #neural_net.add(Dropout(0.2))
    neural_net.summary()

    neural_net.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

    return neural_net

def train(xtrainfinal, ytrainfinal, neural_net):
    history = neural_net.fit(xtrainfinal, ytrainfinal, verbose=1, epochs=10) #TODO: CHANGE?

def test(xtest, ytest, neural_net):
    """Reports the fraction of the test set that is correctly classified.

    The predict() keras method is run on each element of x_test, and the result
    is compared to the corresponding element of y_test. The fraction that
    are classified correctly is tracked and returned as a float.
    """
    loss, accuracy = neural_net.evaluate(xtest, ytest, verbose=0)
    return accuracy

def crossValidation():
    neural_net = settingItUp()
    folds = 5
    # files = []
    # labels = []
    # for j in range(40):
    #     files.append("/scratch/tkyaw1/outfile" + str(j) + ".npz")
    #     labels.append("/scratch/tkyaw1/labels" + str(j) + ".npz")
    # files = np.array(files)
    # labels = np.array(labels)

    filesSmallSubset = "/scratch/tkyaw1/smallSubset.npz"
    labelsSmallSubset = "/scratch/tkyaw1/smallLabels.npz"


    percentlist = []
    for i in range(5):
        b = zeros(40, dtype = bool)
        bcopy = b
        start = i*8
        end = (i+1) * 8
        bcopy[start:end] = True

        xtrain = files[logical_not(bcopy)]
        trainLabels = labels[logical_not(bcopy)]
        xtest = files[bcopy]
        testLabels = labels[bcopy]


        for j in range(len(xtrain)):
            outfile = np.load(xtrain[j])
            loadedOutfile = outfile['arr_0']

            labels = np.load(trainLabels[j])
            loadedLabels = labels['arr_0']

            # training
            train(loadedOutfile, loadedLabels, neural_net)

        foldAccs = []
        for x in range(len(xtest)):
            outfile = np.load(xtest[x])
            loadedOutfile = outfile['arr_0']

            labels = np.load(testLabels[x])
            loadedLabels = labels['arr_0']

            foldAcc = test(loadedOutfile, loadedLabels, neural_net)
            foldAccs.append(foldAcc)

        accuracy = sum(foldAccs)/float(len(foldAccs))
        #testing
        percentlist.append(accuracy)

    average = sum(percentlist)/float(len(percentlist))
    print "average accuracy over folds", average

crossValidation()
