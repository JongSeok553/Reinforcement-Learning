import numpy as np
import csv
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, BatchNormalization, concatenate, Flatten
from keras.optimizers import Adam
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras import backend as K
import random
from collections import deque

class Train:
    def __init__(self):
        self.image_path = 'train_data/image/Test/'
        self.pos_path = 'train_data/image/pos_data/'
        self.save_path = 'train_data/image/save_model/'
        self.file_name = '1000_data_colabo_2'

        self.pos_data = open(self.pos_path + 'heading_data.txt', 'r')
        self.lines = self.pos_data.readlines()
        self.data_length = len(self.lines)

        print("read?", self.data_length)
        self.learning_rate = 0.001
        self.input_shape = 11
        self.output_shape = 7
        self.img_row = 84
        self.img_col = 84

        self.Y_train = []
        self.action_train = np.zeros((self.data_length, 3))
        self.state_train = np.zeros((self.data_length, 3))

        self.model = self.build_model()
        self.hist = []
        self.discount_factor = 0.001
        self.epochs = 100
        self.batch_size = 100
        self.Tau = 0.99
        self.discount_Tau = 0.99
        self.images = []
        self.pos_data = []
        self.read_all()

    def read_all(self):
        files = glob.glob(self.image_path + '*.jpg')
        for file in files:
            self.images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        # self.images = self.images.reshape(self.images.shape[0], self.img_row, self.img_col, 1)
        self.images = np.reshape(self.images, (-1, self.img_col, self.img_row, 1))
        for i in range(len(self.images)):
            self.images[i] = self.images[i] / 255.0

    # def get_action(self, state):
    #     coin = random.randint(0, 100)
    #     if coin <= 5:
    #         action = random.randint(0, 6)
    #         return action
    #     else:
    #         state = np.reshape(state, (-1, 11))
    #         self.prediction = self.model.predict(x=state)
    #         return np.argmax(self.prediction)
    #
    # def replaymemory(self,state, action, reward, next_state):
    #     self.memory.append((state, action, reward, next_state))

    def build_model(self):
        image = Input(shape=(84, 84, 1))
        x = Conv2D(64, (6, 6), strides=(3, 3), activation='relu')(image)
        x = BatchNormalization()(x)
        x = Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        image_dense = Dense(16, activation='relu')(x)

        state = Input(shape=(3,))

        state_vector = Dense(32, activation='relu')(state)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(16, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(8, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        action = concatenate([image_dense, state_vector])

        action = Dense(3, activation='sigmoid')(action)

        accel = Dense(1, activation='sigmoid')(action)
        steer = Dense(1, activation='tanh')(action)
        brake = Dense(1, activation='sigmoid')(action)

        action = concatenate([accel, steer, brake])
        model = Model(inputs=[image, state], outputs=action)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def train_model(self):
        for i in range(self.data_length):
            data = self.lines[i].split('\t')
            self.state_train[i][0] = float(data[0])
            self.state_train[i][1] = float(data[1])
            self.state_train[i][2] = float(data[2])

            self.action_train[i][0] = float(data[3])
            self.action_train[i][1] = float(data[4])
            self.action_train[i][2] = float(data[5])
        self.model.add_loss()
        self.model.input
        self.hist = self.model.fit([self.images, self.state_train], self.action_train, epochs=self.epochs, batch_size=self.batch_size)

    #

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
        plt.savefig(self.save_path + str(self.epochs) + '.png')
        plt.show()
        print(self.file_name + str(self.epochs) + '.png' + ' save plot file')

    # def soft_weight_update(self):
    #
    #     if self.Tau < 0.1:
    #         self.Tau = 0.1
    #     train_network_weight = self.model.get_weights()
    #     target_network_weight = self.target_model.get_weights()
    #     for i in range(len(train_network_weight)):
    #         target_network_weight[i] = (self.Tau * train_network_weight[i]) + ((1-self.Tau) * target_network_weight[i])
    #     self.target_model.set_weights(target_network_weight)
    #     self.Tau = self.Tau * self.discount_Tau
    #     # print("episode ", self.episode, "Score ", self.score, "Tau ", self.Tau, "sub_point ", self.sub_goal_count, self.count)

if __name__ == '__main__':
    train = Train()
    train.train_model()

    train.model.save(train.save_path + train.file_name + str(train.epochs) + '.h5')
    print(train.file_name + str(train.epochs) + '.h5' + " save weight file")
    train.result_plot()
