from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model, model_from_json
from collections import deque
import numpy as np
import random
import tensorflow as tf
import sys
from keras.models import load_model

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

config = tf.ConfigProto()
session = tf.Session(config=config)
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True

class TAgent:
    def __init__(self):
        self.steering = []
        self.throttle = []
        self.brake = []
        self.discount_factor = 0.99
        self.learning_rate = 0.005
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 500
        self.input_shape = (50, 80, 1)
        self.prediction = []
        self.memory = deque(maxlen=1000)    # default 632 byte
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.y = np.zeros((self.batch_size, 4))
        # self.model.load_weights("model_save/1_model.h5")
        # self.target_model.load_weights("model_save/1_model.h5")
        # print("load model", self.model.load_weights)

        # if not self.model.load_weights:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     self.model = self.build_model()
        #     self.target_model = self.build_model()
        #     print("generate new model ")

        #
        # self.update_target_model()

    def replaymemory(self, state, action1, action2,reward, nextstate):
        self.memory.append((state, action1, action2, reward, nextstate))
        # print(self.memory)
        # print("==============================================================")
        # print('sys.getsizeof(x): {0} bytes'.format(sys.getsizeof(self.memory)))
        # print(len(self.memory))+

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0, 5), random.randrange(0, 5)
        else:
            state = np.reshape(state,(-1, 50, 80, 1))
            state = state / 255.0
            self.prediction = self.model.predict(x=state)
            # print("prediction", self.prediction)
            self.prediction = self.prediction.argsort()
            # print("prediction", self.prediction, self.prediction[0][3], self.prediction[0][2] )
            # print("prediction", self.prediction, self.prediction[0][4], self.prediction[0][3])
            # print("prediction", self.prediction)
            return self.prediction[0][4], self.prediction[0][3]
    #
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(6, 6),strides=(3, 3), input_shape=(50, 80, 1), activation='relu'))
        # model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        # model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu'))
        # model.add(BatchNormalization())

        model.add(Flatten())  # 11264
        # model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))

        model.add(Dense(5, activation='sigmoid'))



        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Flatten())                # 11264
        # model.add(Dense(64, activation='relu'))
        # model.add(BatchNormalization())
        #
        # model.add(Dense(5, activation='sigmoid'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def model_train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)
        state = np.zeros((self.batch_size, 50, 80, 1))
        next_state = np.zeros((self.batch_size, 50, 80, 1))

        rewards, action1, action2  = [], [], []
        for i in range(self.batch_size):
            state[i] = np.float32(mini_batch[i][0]/255.0)
            action1.append(mini_batch[i][1])
            action2.append(mini_batch[i][2])
            rewards.append(mini_batch[i][3])
            next_state[i] = np.float32(mini_batch[i][4]/255.0)

        self.y = self.model.predict(x=state, batch_size=self.batch_size)
        target_value = self.target_model.predict(x=next_state, batch_size=self.batch_size)
        target_value_number = target_value.argsort()
        for i in range(self.batch_size):
            self.y[i][int(action1[i])] = rewards[i] + self.discount_factor * (target_value[i][target_value_number[i][4]])
            self.y[i][int(action2[i])] = rewards[i] + self.discount_factor * (target_value[i][target_value_number[i][3]])

        self.model.fit(state, self.y, batch_size=self.batch_size)
        print("finish train")

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("update target model !!")

