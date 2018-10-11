#!/usr/bin/env python3

import sys
import asyncio
import threading
import tensorflow as tf
from tensorflow import keras
import numpy as np

#model made for MCTS estimation
#Qtable functions have just been commented out
#I'll uncomment and port as needed
class MModel:
    def __init__(self, input_shape, num_actions, alpha=0.001, model=None, width=256):
        self.alpha = alpha
        #needs to be saved so we can clone
        self.input_shape = input_shape
        self.num_actions = num_actions
        if model == None:
            #simple feedforward
            inputs = keras.Input(input_shape)
            x = keras.layers.Dense(width, activation='relu')(inputs)
            y = keras.layers.Dense(width, activation='relu')(x)
            probabilityPrediction = keras.layers.Dense(num_actions,
                    activation='softmax', name='prob')(y)
            valuePrediction = keras.layers.Dense(1,
                    activation='tanh', name='value')(y)
            self.model = keras.Model(inputs=inputs,
                    outputs=[probabilityPrediction, valuePrediction])
            """
            #fully connected
            inputs = keras.Input(input_shape)
            x = keras.layers.Dense(width, activation='relu')(inputs)
            x = keras.layers.Dense(width, activation='relu')(x)
            probabilityPrediction = keras.layers.Dense(num_actions,
                    activation='softmax')(y)
            valuePrediction = keras.layers.Dense(1,
                    activation='tanh', name='value')(y)
            self.model = keras.Model(inputs=inputs,
                    outputs=[probabilityPrediction, valuePrediction])
            """

            self._compile()

        else:
            self.model = model

        #used for batched training
        self.savedData = []
        self.savedProb = []
        self.savedValue = []

        loop = asyncio.get_event_loop()
        #inputs go here
        self.evalQueue= asyncio.Queue(loop=loop)
        self.evalLock = asyncio.Lock()

        self.dataLabelLock = asyncio.Lock()


    #compiles the model, which needs to be done both on init and on load
    def _compile(self):
        #have two separate outputs, which use separate errors
        lossMap = {'prob': 'categorical_crossentropy',
                'value': 'mse'}
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.alpha),
                loss=lossMap)

    #updates alpha for the model, recompiling if necessary
    def setAlpha(self, alpha):
        self.alpha = alpha
        self._compile()

    #uses the model to get the predicted label for the piece of data
    def getValue(self, data):
        output = self.model.predict(np.array([data]))
        prediction = {
            'prob': output[0][0],
            'value': output[1][0],
        }
        return prediction

    #actual loop for async evaluating inputs
    async def autoEval(self):
        while True:
            data = []
            results = []
            #wait until we get an input
            datum, result = await self.evalQueue.get()
            data.append(datum)
            results.append(result)
            #consume all inputs
            #queue isn't threadsafe, so have to use a lock
            async with self.evalLock:
                while not self.evalQueue.empty():
                    datum, result = self.evalQueue.get_nowait()
                    data.append(datum)
                    results.append(result)
            #evaluate all the inputs at once
            output = await self.getValues(data)
            #send the data back via futures
            for i in range(len(output)):
                results[i].set_result(output[i])

    #gets a value on the next evaluation
    async def asyncGetValue(self, data):
        async with self.evalLock:
            loop = asyncio.get_event_loop()
            #future for the result of this particular evaluation
            future = loop.create_future()
            await self.evalQueue.put((data, future))
        return await future

    #like getValue(), except data is an array
    #returns the corresponding output array
    #each entry in the return value is a dict
    async def getValues(self, data):
        output = self.model.predict(np.array(data))
        probs = output[0]
        values = output[1]
        predictions = []
        for i in range(len(probs)):
            predictions.append({
                'prob': probs[i],
                'value': values[i],
            })
        return predictions

    #saves the data-label for a later batched update
    def addDataLabel(self, data, prob, value):
        self.savedData.append(data)
        self.savedProb.append(prob)
        self.savedValue.append(value)

    #async version of addDataLabel
    #synchronization has overhead, so this is batched
    async def addDataLabels(self, dataLabels):
        async with self.dataLabelLock:
            for data, prob, value in dataLabels:
                self.savedData.append(data)
                self.savedProb.append(prob)
                self.savedValue.append(value)

    #updates the model to try to fit the data point
    #def updateTable(self, data, label):
        #self.model.fit(np.array([data]), np.array([label]), verbose=0)

    def batchedUpdate(self, epochs=1, batch_size=None):
        if len(self.savedData) > 0:
            labels = {'prob': np.array(self.savedProb),
                    'value': np.array(self.savedValue)}
            self.model.fit(np.array(self.savedData), labels, verbose=0,
                    epochs=epochs, batch_size=batch_size)
            self.savedData = []
            self.savedProb = []
            self.savedValue = []

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



class QModel:
    def __init__(self, input_shape, alpha=0.001, model=None, width=256):
        self.alpha = alpha
        #needs to be saved so we can clone
        self.input_shape = input_shape
        if model == None:
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
            """

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

