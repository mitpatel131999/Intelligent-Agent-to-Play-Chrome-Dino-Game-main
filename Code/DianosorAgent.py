import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf


class DianosorAgent:    # agent actions or operations will done here
    def __init__(self): # the below are the standard neural network models
        diano_model = Sequential()
        diano_model.add(Conv2D(32, (8, 8), input_shape=(76, 384, 4), strides=(2, 2), activation='relu'))
        diano_model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
        diano_model.add(Conv2D(64, (4, 4), activation='relu', strides=(1, 1)))
        diano_model.add(MaxPooling2D(pool_size=(7, 7), strides=(3, 3)))
        diano_model.add(Conv2D(128, (1, 1), strides=(1, 1), activation='relu'))
        diano_model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
        diano_model.add(Flatten())
        diano_model.add(Dense(384, activation='relu'))
        diano_model.add(Dense(64, activation="relu"))
        diano_model.add(Dense(8, activation="relu"))
        diano_model.add(Dense(3, activation="linear"))

        diano_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

        self.diano_model = diano_model

        self.diano_model.load_weights("DinoTrain.h5")

        self.model_memory = []

        self.data_screen = []
        self.data_action = []

        self.location = 0

    def action(self, state):    # to perform actions by agent

        stateConv = state

        Q_value = self.diano_model.predict(np.reshape(stateConv, (1, 76, 384, 4)))

        probablity = tf.nn.softmax(tf.math.divide((Q_value.flatten()), 1))

        action = np.random.choice(range(3), p=np.array(probablity))

        return action

    def remember(self, state, ns, action, reward, done, location):  # this will track the location or state of the agent

        self.location = location

        tamp = np.array([state, ns, action, reward, done])

        self.model_memory.append(tamp)

    def tream_memory(self): # the memory checking of batch of experiences

        self.batchSize = 256

        if len(self.model_memory) > 50000:
            self.model_memory = []

        if len(self.model_memory) < self.batchSize:
            return True

        return False

    def learn(self):    # this function will help agent to learn the game

        self.sample_Size = 256

        if self.tream_memory():
            return

        sample = np.array(random.sample(self.model_memory, self.sample_Size))

        alpha = 0.9

        actions = sample[:, 2].reshape(self.sample_Size).tolist()

        rewards = sample[:, 3].reshape(self.sample_Size).tolist()

        Predict = sample[:, 0].reshape(self.sample_Size).tolist()

        nextPredict = sample[:, 1].reshape(self.sample_Size).tolist()

        pr = self.diano_model.predict(np.reshape(Predict, (self.sample_Size, 76, 384, 4)))

        npr = self.diano_model.predict(np.reshape(Predict, (self.sample_Size, 76, 384, 4)))

        pr = np.array(pr)  # all predicted values

        npr = np.array(npr)

        for i in range(self.sample_Size):
            action = actions[i]
            reward = rewards[i]
            ns = npr[i]  # storing next predicted value to next state
            q = pr[i, action]   # then getting q value
            if reward < -25:    # calculating reward
                pr[i, action] = reward
            else:
                pr[i, action] += alpha * (reward + alpha * np.max(ns) - q)

        self.data_screen.append(np.reshape(Predict, (self.sample_Size, 76, 384, 4)))
        self.data_action.append(pr)
        history = self.diano_model.fit(self.data_screen, self.data_action, batch_size=5, epochs=1, verbose=0)

        self.data_screen = []
        self.data_action = []
