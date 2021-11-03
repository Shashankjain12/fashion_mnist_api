import argparse
# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-t", "--train",required = True, help = "Pass the training dataset location")
parser.add_argument("-ts", "--test",required = True, help = "Pass the testing dataset location")
# Read arguments from command line
args = parser.parse_args()

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten


# Mnist reader to read images from the gzip file present in the training dataset
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)

    return images, labels

def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# Training and testing set splitting and using of data
X_train, y_train = load_mnist(args.train, kind='train')
X_test, y_test = load_mnist(args.test, kind='t10k')
# Making the y_train and y_test to categorical values as they are ranging from 0 to 9
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Normalizing the pixel values from 0 to 1 range only
X_train, X_test = prep_pixels(X_train, X_test)


# Initializing the CNN
classifier = Sequential()
# Adding first Convolution layer adding 32 as feature detector
# With 3 * 3 as stride with the input shape as 28 * 28 and activation as relu to increase non linearity between data
classifier.add(Convolution2D(32, (3, 3), padding = "same", input_shape = (28 ,28, 1), activation = 'relu'))
# Adding second layer of maxpooling to extract the best features from the data with stride of 2*2
classifier.add(MaxPooling2D( pool_size = (2, 2)))
# Now then flatten all of the data to pass it to the ANN to classify the data
classifier.add(Flatten())
# Full connection by adding the Dense layer to the model by adding 128 nodes as an output to the first layer
classifier.add(Dense(units = 128, activation = 'relu'))
# Adding the hidden layer to the model with output dimension as 10 as we can get 10 distinct set of values
classifier.add(Dense(units = 10, activation = 'softmax'))
# Now compiling the whole classifier which configures the whole model for training with loss as categorical crossentropy
# And using Stochastic Gradient Decent with learning rate as 0.01
from tensorflow.keras.optimizers import SGD
opt = SGD(learning_rate=0.01, momentum=0.9)
# classifier.compile(loss = "categorical_crossentropy", optimizer = opt)
classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Now training our data set with batch size as 600 to train data individually and
# epochs as 100 to train the data 100 times
# Validating the training on X_test and y_test for the trained data
# Steps per epoch will be running 100 batches per epoch
classifier.fit(
    x = X_train,
    y = y_train,
    batch_size = 600,
    epochs = 100,
    verbose = "auto",
    validation_data = (X_test, y_test),
    shuffle=True,
    initial_epoch=0,
    steps_per_epoch = 100,
    validation_steps= 100,
    validation_batch_size = 100,
    validation_freq=1,
    max_queue_size=10,
    workers = 2,
    use_multiprocessing = True,
)

classifier.save("/home/shashank/Desktop/vectorai/saved_model/fashion_mnist.h5")

