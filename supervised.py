import numpy as np
import csv
from keras.models import Sequential, save_model
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras import metrics
from keras import models
import numpy as np
import matplotlib.pyplot as plt

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
        self.data_file_path = 'model_supervised/'
        self.f = open(self.data_file_path + 'heading_data.txt', 'r')
        self.lines = self.f.readlines()
        self.data_length = len(self.lines)
        self.learning_rate = 0.001
        self.input_shape = 3
        self.output_shape = 3
        self.X_train = np.zeros((self.data_length, self.input_shape))
        self.Y_train = np.zeros((self.data_length, self.output_shape))
        self.accel = []
        self.steer = []
        self.brake = []
        self.model = self.build_model()
        self.hist = 0
        self.epochs = 10000
        self.batch_size = 1000

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.input_shape, ), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
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
            self.X_train[i][2] = float(data[2])

            self.Y_train[i][0] = float(data[3])
            self.Y_train[i][1] = float(data[4])
            self.Y_train[i][2] = float(data[5])
            # print(self.Y_train[i][0])

        self.hist = self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size)

    def result_plot(self):
        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()

        loss_ax.plot(self.hist.history['loss'], 'y', label='train loss')

        # acc_ax.plot(self.hist.history['acc'], 'b', label='train acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        # acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        # acc_ax.legend(loc='lower left')
        plt.savefig('model_supervised/plot/1000_batch_heading_Test' + str(self.epochs) + '.png')
        plt.show()
        print('1000_batch_heading_Test' + str(self.epochs) + '.png' + ' save plot file')


if __name__ == '__main__':
    train = Train()
    train.train_model()
    train.model.save('model_supervised/' + 'TTTTTTTTTTTTt' + str(train.epochs) + '.h5')
    print('1000_batch_heading_Test' + str(train.epochs) + '.h5' + " save weight file")
    train.result_plot()
