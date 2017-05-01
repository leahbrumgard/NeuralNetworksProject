import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from numpy import argmax

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D, Flatten

# TODO: load test file (if you want)
# TODO: normalize everything!!!!!!!!!!!!! TODO TODO TODO  TODO
# NOTE: relu for non output layers, sigmoid for output layers ?? not sure y
# NOTE: adding another pair of pooling/conv layer dropped accuracy a lot

def main():
    #y_train_vectors = to_categorical(y_train, num_categories)
    #y_test_vectors = to_categorical(y_test, num_categories)


    xtrain1 = np.load('/scratch/tkyaw1/sampleSubset.npz')
    ytrain1 = np.load('/scratch/tkyaw1/sampleLabels.npz')

    # x_test_images = (x_test.reshape(10000, 28, 28, 1) - x_min) / float(x_max - x_min)

    neural_net = Sequential()

    neural_net.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))
    neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))

    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))
    neural_net.add(Conv2D(32, (3, 3), activation = 'relu'))

    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))
    neural_net.add(Conv2D(32, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))

    neural_net.add(Flatten())
    neural_net.add(Dense(32, activation = 'relu'))
    neural_net.add(Dropout(0.5))
    neural_net.add(Dense(2, activation = 'sigmoid'))
    #neural_net.add(Dropout(0.2))


    """
    #####
    neural_net = Sequential()

    neural_net.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))
    neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))
    neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))

    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))
    neural_net.add(Dropout(0.25))

    neural_net.add(Conv2D(32, (3, 3), activation = 'relu'))
    neural_net.add(Conv2D(32, (3, 3), activation = 'relu'))

    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (500, 500, 3)))
    neural_net.add(Dropout(0.25))

    neural_net.add(Flatten())
    neural_net.add(Dense(32, activation = 'relu'))
    neural_net.add(Dropout(0.5))
    neural_net.add(Dense(2, activation = 'sigmoid'))
    #neural_net.add(Dropout(0.2))
    #####
    """

    neural_net.summary()

    xtrainfinal = xtrain1['arr_0']
    ytrainfinal = ytrain1['arr_0']
    x_max = xtrainfinal.max()
    x_min = xtrainfinal.min()
    xtrainfinal = (xtrainfinal - float(x_min)) / float(x_max - x_min)

    neural_net.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    history = neural_net.fit(xtrainfinal, ytrainfinal[:1000,:], verbose=1, epochs=10) #validation_data=(xtrainfinal, ytrainfinal[:1000,:]),
    loss, accuracy = neural_net.evaluate(xtrainfinal, ytrainfinal[:1000,:], verbose=0)
    #plt.imshow(xtrainfinal[0])
    #plt.show()
    print "correct answer:", ytrainfinal[0]
    prediction = neural_net.predict(xtrainfinal)
    print "prediction:", prediction
    #print "nn output", argmax(neural_net.predict(xtrainfinal)[0])
    print "accuracy: {}%".format(accuracy*100)

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
#,
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
