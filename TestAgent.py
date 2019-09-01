from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from collections import deque
import numpy as np
import random
import tensorflow as tf
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class TAgent:
    def __init__(self):
        self.steering = []
        self.throttle = []
        self.brake = []
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 500
        self.input_shape = (480, 270 ,3)
        self.prediction = []
        self.memory = deque(maxlen=1000)    # default 632 byte

        self.model = self.build_model()
        self.target_model = self.build_model()
        #
        # self.update_target_model()

    def replaymemory(self, state, reward, nextstate):
        self.memory.append((state, reward, nextstate))
        # print(self.memory)
        # print("==============================================================")
        # print('sys.getsizeof(x): {0} bytes'.format(sys.getsizeof(self.memory)))
        # print(len(self.memory))+

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # th = random.randrange(0, 4)
            # st = random.randrange(0, 2)
            # br = random.randrange(0, 2)
            # # print("random action", th, st, br)
            # print("random action")
            return random.randrange(0, 4)
        else:
            # print("model predict action")
            state = np.reshape(state, (-1, 480, 270, 3))
            self.prediction = self.model.predict(state)
            # self.throttle.append((predict[0][0], predict[0][1]))
            # self.steering.append((predict[0][2], predict[0][3]))
            # self.brake.append((predict[0][4], predict[0][5]))
            # print("prediction",predict, np.argmax(predict[0]))
            # print("model action",self.throttle[0], self.steering[0], self.brake[0], np.argmax(self.throttle[0]), np.argmax(self.steering[0]), np.argmax(self.brake[0]))
            # print(self.throttle[0], self.steering[0], self.brake[0])
            # print(np.argmax(self.throttle[0]), np.argmax(self.steering[0]), np.argmax(self.brake[0]))
            # return np.argmax(self.throttle[0]), np.argmax(self.steering[0]), np.argmax(self.brake[0])
            return np.argmax(self.prediction[0])
    #
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(480, 270, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(4, activation='sigmoid'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def model_train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        mini_batch = random.sample(self.memory, self.batch_size)
        state = np.zeros((self.batch_size, 480, 270, 3))
        next_state = np.zeros((self.batch_size, 480, 270, 3))
        target = np.zeros((self.batch_size, 4))
        rewards = []

        for i in range(self.batch_size):
            state[i] = np.float32(mini_batch[i][0]/255.)
            rewards.append(mini_batch[i][1])
            next_state[i] = np.float32(mini_batch[i][2]/255.)

        target_value = self.target_model.predict(x=next_state,batch_size=self.batch_size)

        # print(np.argmax(self.prediction))
        # print("================================")
        for i in range(self.batch_size):
            target[i][np.argmax(self.prediction[0])] = rewards[i] + self.discount_factor * (max(target_value[0]))

            # target[i][2 + steering[i]] = rewards[i] + self.discount_factor * (
            #     max(target_value[0][2], target_value[0][3]))
            #
            # target[i][4 + brake[i]] = rewards[i] + self.discount_factor * (
            #     max(target_value[i][4], target_value[i][5]))
        # print("target value ", target)
        # print(self.model.predict(state))
        # print("========================")
        # print(target_value)
        # print("========================")
        # print(target)
        self.model.fit(state, target, batch_size=self.batch_size)
        print("finish train")

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("update target model !!")

















