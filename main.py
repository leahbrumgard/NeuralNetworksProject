import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from numpy import argmax, zeros, logical_not

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D, Flatten

# TODO: load test file (if you want)
# TODO: normalize everything!!!!!!!!!!!!! TODO TODO TODO  TODO
# NOTE: relu for non output layers, sigmoid for output layers ?? not sure y
# NOTE: adding another pair of pooling/conv layer dropped accuracy a lot

def train(xtrainfinal, ytrainfinal):
    #y_train_vectors = to_categorical(y_train, num_categories)
    #y_test_vectors = to_categorical(y_test, num_categories)


    # xtrain1 = np.load('/scratch/tkyaw1/sampleSubset.npz')
    # ytrain1 = np.load('/scratch/tkyaw1/sampleLabels.npz')

    # x_test_images = (x_test.reshape(10000, 28, 28, 1) - x_min) / float(x_max - x_min)

    neural_net = Sequential()

    neural_net.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))
    neural_net.add(Conv2D(128, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')) #WHY WE DO THIS?

    neural_net.add(Conv2D(128, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    # neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))
    # neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))

    # neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))
    # neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))

    neural_net.add(Flatten())
    neural_net.add(Dense(32, activation = 'relu'))
    neural_net.add(Dropout(0.5))
    neural_net.add(Dense(2, activation = 'sigmoid'))
    #neural_net.add(Dropout(0.2))
    neural_net.summary()

    neural_net.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    history = neural_net.fit(xtrainfinal, ytrainfinal, verbose=1, epochs=1) #validation_data=(xtrainfinal, ytrainfinal[:1000,:]),

    return neural_net
    #plt.imshow(xtrainfinal[0])
    #plt.show()
    print "correct answer:", ytrainfinal[0]
    prediction = neural_net.predict(xtrainfinal)
    print "prediction:", prediction
    print "accuracy: {}%".format(accuracy*100)

def test(xtest, ytest, neural_net):
    """Reports the fraction of the test set that is correctly classified.

    The predict() keras method is run on each element of x_test, and the result
    is compared to the corresponding element of y_test. The fraction that
    are classified correctly is tracked and returned as a float.
    """
    # correct = 0
    # labels = neural_net.predict(xtest)
    # for i in range(len(ytest)):
    #     if labels[i] == ytest[i]:
    #         correct += 1
    loss, accuracy = neural_net.evaluate(xtest, ytest, verbose=0)

    return accuracy

def crossValidation():
    folds = 5
    # files = []
    # labels = []
    # for j in range(20):
    #     files.append("/scratch/tkyaw1/outfile" + str(j) + ".npz")
    #     labels.append("/scratch/tkyaw1/labels" + str(j) + ".npz")
    # files = np.array(files)
    # labels = np.array(labels)

    # filesSmallSubset = "/scratch/tkyaw1/smallSubset.npz"
    # labelsSmallSubset = "/scratch/tkyaw1/smallLabels.npz"
    xtrain1 = np.load('/scratch/tkyaw1/sampleSubset.npz')
    ytrain1 = np.load('/scratch/tkyaw1/sampleSubset.npz')
    xtrainfinal = xtrain1['arr_0']
    ytrainfinal = ytrain1['arr_0']
    x_max = xtrainfinal.max()
    x_min = xtrainfinal.min()
    xtrainfinal = (xtrainfinal / float(255))

    percentlist = []
    for i in range(5):
        # b = zeros(20, dtype = bool)
        b = zeros(1000, dtype = bool)
        bcopy = b
        bcopy[i*4: (i+1)*4] = True
        xtrain = xtrainfinal[logical_not(bcopy)]
        trainLabels = ytrainfinal[logical_not(bcopy)]
        xtest = xtrainfinal[bcopy]
        testLabels = ytrainfinal[bcopy]

        # training
        neural_net = train(xtrainfinal, ytrainfinal)

        #testing
        accuracy = test(xtest, testLabels, neural_net)
        percentlist.append(accuracy)

    average = sum(percentlist)/float(len(percentlist))
    print "average accuracy over folds", average

crossValidation()
