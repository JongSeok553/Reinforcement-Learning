import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras import metrics
import numpy as np
import random
import tensorflow as tf
# f = open('data.txt','r')
# lines = f.readlines()
# # print(lines)
# # line = lines[0].split('\t')
# # print(ss[0], ss[1], ss[2], ss[3], ss[4])
# data_length=len(lines)
# # print(data_length)
# xy = np.zeros((data_length, 2))
#
# accel = []
# steer = []
# brake = []
#
# for i in range(len(lines)):
#     data = lines[i].split('\t')
#     xy[i][0] = float(data[0])
#     xy[i][1] = float(data[1])
#     accel.append(data[2])
#     steer.append(data[3])
#     brake.append(data[4])
# print(xy)
class Train:
    def __init__(self):
        self.f = open('data.txt', 'r')
        self.lines = self.f.readlines()
        self.data_length = len(self.lines)
        self.learning_rate = 0.001
        self.batch_size = 128
        self.input_shape = 2
        self.output_shape = 3
        self.X_train = np.zeros((self.data_length, self.input_shape))
        self.Y_train = np.zeros((self.data_length, self.output_shape))
        self.accel = []
        self.steer = []
        self.brake = []
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.input_shape, ), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.output_shape, activation='tanh'))
        # accel, brake는 0이하는 0으로 처리해야할듯..
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def train_model(self):
        for i in range(self.data_length):
            data = self.lines[i].split('\t')
            self.X_train[i][0] = float(data[0])
            self.X_train[i][1] = float(data[1])

        for i in range(self.data_length):
            self.Y_train[i][0] = float(data[2])
            self.Y_train[i][1] = float(data[3])
            self.Y_train[i][2] = float(data[4])

        self.model.fit(self.X_train, self.Y_train, epochs=10000, batch_size=self.batch_size)

if __name__ == '__main__':
    train = Train()
    train.train_model()