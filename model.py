#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np

class QModel:
    def __init__(self, input_shape, alpha=0.001, model=None, width=256):
        self.alpha = alpha
        #needs to be saved so we can clone
        self.input_shape = input_shape
        if model == None:
            """
            #simple feedforward
            inputs = keras.Input(input_shape)
            x = keras.layers.Dense(width, activation='relu')(inputs)
            y = keras.layers.Dense(width, activation='relu')(x)
            predictions = keras.layers.Dense(1)(y)
            self.model = keras.Model(inputs=inputs, outputs=predictions)
            """
            #fully connected
            inputs = keras.Input(input_shape)
            x = keras.layers.Dense(width, activation='relu')(inputs)
            x = keras.layers.Dense(width, activation='relu')(x)
            predictions = keras.layers.Dense(1)(x)
            self.model = keras.Model(inputs=inputs, outputs=predictions)

            self._compile()

        else:
            self.model = model

        #used for batched training
        self.savedData = []
        self.savedLabels = []


    #compiles the model, which needs to be done both on init and on load
    def _compile(self):
        """
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.alpha),
                loss='mse',
                metrics=['mae'])
        """
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.alpha),
                loss='logcosh')

    #updates alpha for the model, recompiling if necessary
    def setAlpha(self, alpha):
        self.alpha = alpha
        self._compile()

    #uses the model to get the predicted label for the piece of data
    def getValue(self, data):
        return self.model.predict(np.array([data]))[0][0]

    #like getValue(), except data is an array
    #returns the corresponding output array
    def getValues(self, data):
        #output only has one value
        return [v[0] for v in self.model.predict(np.array(data))]

    #saves the data-label for a later batched update
    def addDataLabel(self, data, label):
        self.savedData.append(data)
        self.savedLabels.append(label)

    #updates the model to try to fit the data point
    def updateTable(self, data, label):
        self.model.fit(np.array([data]), np.array([label]), verbose=0)

    def batchedUpdate(self, epochs=1, batch_size=None):
        if len(self.savedData) > 0:
            self.model.fit(np.array(self.savedData), np.array(self.savedLabels), verbose=0,
                    epochs=epochs, batch_size=batch_size)
            self.savedData = []
            self.savedLabels = []

    #saves the model to the file name (which might include path)
    def saveModel(self, name):
        self.model.save(name, include_optimizer=False)

    #loads the model from the file name (which might include path)
    def loadModel(self, name):
        self.model = keras.models.load_model(name)
        self._compile()

    #returns another instance of QModel with a clone of the current model
    #this should be a deep copy
    def clone(self):
        return QModel(input_shape=self.input_shape,
                model=keras.models.clone_model(self.model),
                alpha=self.alpha)

if __name__ == '__main__':
    data = np.array([
        [-1,-1],
        [-1,1],
        [1,-1],
        [1,1],
    ])

    labels = np.array([
        0,
        2,
        2,
        0,
    ])

    model = QModel(input_shape=(2,), alpha=0.001)
    for i in range(100):
        for d in range(len(data)):
            model.updateTable(data[d], labels[d])
    for i in range(4):
        print(data[i], model.getValue(data[i]))

