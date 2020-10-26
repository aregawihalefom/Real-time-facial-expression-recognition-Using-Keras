import numpy as np
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.layers import *


class Network1:

    def __init__(self, num_classes, desc=None):
        self.num_classes = num_classes
        self.desc = desc
        self.model = None

    def create_model(self):
        model = Sequential()

        # convolutional layer 1
        model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        # Convolutional Layer 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Convolutional Layer 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))

        # Flattening
        model.add(Flatten())

        # Fully connected layer 1st layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        # Fully connected layer 2nd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(self.num_classes, activation='sigmoid'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        self.model = model

        return model
