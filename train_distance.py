from keras.models import Sequential, save_model, Model, load_model
from keras.layers import Dense, Lambda, Input, Add, Average, Subtract, average, BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from collections import deque

import numpy as np
import math
import random
import tensorflow as tf
import time
import keras.backend as K

class Distance:
    def __init__(self):
        self.episode = 0
        self.data_file_path = 'model_supervised/'
        self.train_data_path = 'DDDQN/'
        self.f = open(self.data_file_path + 'heading_data.txt', 'r')
        self.lines = self.f.readlines()
        self.data_length = len(self.lines)
        self.point = []
        self.distance = 0
        self.memory = deque(maxlen=2000)
        self.make_point()
        self.x = self.point[0][0]
        self.y = self.point[0][1]
        self.h = self.point[0][2]
        self.x2 = self.point[1][0]
        self.y2 = self.point[1][1]
        self.h2 = self.point[1][2]
        self.x3 = self.point[2][0]
        self.y3 = self.point[2][1]
        self.h3 = self.point[2][2]

        self.sub_goal = []
        self.sub_goal_count = 0
        self.dest = 0
        self.pre_dest = 0

        self.seg_a = 0
        self.seg_b = 0
        self.seg_c = 0
        self.count = 0
        self.cur_x = 0
        self.cur_y = 0
        self.cur_h = 0
        self.spawn_x = 0
        self.spawn_y = 0
        self.spawn_h = 0
        self.spawn_flag = False
        self.last_p = np.shape(self.point)[0] - 1

        model_path = 'model_supervised/4action/'
        model_file = '20004action_xyhd_all' + '.h5'
        model_name = model_path + model_file
        model = load_model(model_name)
        self.model = model
        self.target_model = model
        self.input_shape = 4
        self.output_shape = 3
        self.epochs = 10
        self.batch_size = 300
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.Tau = 0.01
        self.train_start = False
        self.prediction =[]
        self.angle = 0
        self.d = 0
        self.end_train = False


        self.train_data_length = 0
        self.state = np.zeros((self.train_data_length, self.input_shape))
        self.next_state = np.zeros((self.train_data_length, self.input_shape))
        self.action = np.zeros((self.train_data_length, 2))
        self.steer = np.zeros((self.train_data_length, 2))
        self.re = np.zeros((self.train_data_length, 1))
        self.ac = np.zeros((self.train_data_length, 1))

        self.gradient = 0
        self.reciprocal = 0
        self.episilon = 0.000001
        self.random_prob = 0.9
        self.random_dicount = 0.9

    def done(self):
        self.spawn_flag = True
        self.x = self.point[self.count][0]
        self.y = self.point[self.count][1]
        self.h = self.point[self.count][2]
        self.spawn_x = self.point[1][0]
        self.spawn_y = self.point[1][1]
        self.spawn_h = self.point[1][2]
        # self.data_file.close()

    def next_point(self):
        self.x = self.point[self.count][0]
        self.y = self.point[self.count][1]
        self.h = self.point[self.count][2]

        self.x2 = self.point[self.count + 1][0]
        self.y2 = self.point[self.count + 1][1]
        self.h2 = self.point[self.count + 1][2]

        self.x3 = self.point[self.count + 2][0]
        self.y3 = self.point[self.count + 2][1]
        self.h3 = self.point[self.count + 2][2]
        # self.distance = math.sqrt(math.pow((self.x - self.cur_x), 2) + math.pow((self.y - self.cur_y), 2))
        # print("next point", self.count, self.d)

    def make_point(self):
        for i in range(self.data_length):
            if i%100 == 0:
                self.sub_goal.append((float(data[0]), float(data[1])))

            data = self.lines[i].split('\t')
            self.point.append((float(data[0]), float(data[1]), float(data[2])))

    def next(self, cur_x, cur_y):
        if self.count + 1 > self.data_length:
            self.done()
        else:
            # if self.gradient <= 0:
            #     self.reciprocal = 1 / (self.gradient + self.episilon)
            # else:
            #     self.reciprocal = -1 / (self.gradient + self.episilon)
            #
            # c = cur_y - (self.reciprocal * cur_x)
            # # print("reciprocal", self.reciprocal)
            d1 = math.sqrt(math.pow((self.x - cur_x), 2) + math.pow((self.y - cur_y), 2))
            d2 = math.sqrt(math.pow((self.x2 - cur_x), 2) + math.pow((self.y2 - cur_y), 2))
            p3_p2 = math.sqrt(math.pow((self.x3 - self.x2), 2) + math.pow((self.y3 - self.x2), 2))
            p3_p1 = math.sqrt(math.pow((self.x3 - self.x), 2) + math.pow((self.y3 - self.x), 2))
            # self.count += 1
            # self.next_point()
            # print("x1, x2,",self.x, self.x2, "y1, y2,",self.y, self.y2, "d1, d2, ",d1, d2, "p3_p2", p3_p2, "p3_p1", p3_p1)
            if d1 >= d2:
                self.count += 1
                # print("d1d2")
                self.next_point()
            else:
                if p3_p2 >= p3_p1:
                    self.count += 1
                    # print("p3p2")
                    self.next_point()

    def get_dest(self, curx, cury):
        sub_x = self.sub_goal[self.sub_goal_count][0]
        sub_y = self.sub_goal[self.sub_goal_count][1]

        self.dest = math.sqrt((sub_x - curx)**2 + (sub_y - cury)**2)
        if (self.pre_dest - self.dest) > 0.1:
            if self.d < 1:
                reward = 0.1
        else:
            reward = -0.2

        if self.dest < 1:
            reward += 1
            self.sub_goal_count += 1

        self.pre_dest = self.dest

        return self.dest, reward

    def cal_reward(self, d, velocity):
        action_reward = 0
        if abs(d) < 0.1:
            reward = -1#(abs(0.1 - d)) * 0.01
        else:
            reward = 1#-(abs(0.1 - d))

        if velocity < 1:
            action_reward = -1
        else:
            action_reward = 0.01

        return reward, action_reward

    def get_d(self, cur_x, cur_y, cur_h):
        self.cur_x = cur_x
        self.cur_y = cur_y
        self.cur_h = cur_h
        b = self.y2 - self.y
        a = self.x2 - self.x
        c = (-b * self.x) + (a * self.y)
        self.gradient = a / (b + self.episilon)
        self.next(cur_x, cur_y)
        self.dest, re = self.dest(self.cur_x, self.cur_y)


        if a == 0:
            self.d = 0 #abs(cur_y - self.y2)
        elif b == 0:
            self.d = 0 #abs(cur_x - self.x2)
        elif a == 0 and b == 0:
            self.d = 0
        else:
            self.d = abs(b*cur_x + (-a * cur_y) + c) / (math.sqrt((math.pow(a, 2)) + (math.pow(b, 2))) + self.episilon)
        print("ddd", self.d)
        return self.d, self.dest, re

    def replaymemory(self, state, action1, action2,reward, nextstate):
        self.memory.append((state, action1, action2, reward, nextstate))


    def respawn(self, x, y, h, col):
        if self.d > 2:
            self.count = 0
            self.spawn_x = self.point[self.sub_goal_count][0]
            self.spawn_y = self.point[self.sub_goal_count][1]
            self.spawn_h = self.point[self.sub_goal_count][2]
            self.spawn_flag = True
            # self.get_d(self.spawn_x, self.spawn_y,self.spawn_h)
            # self.next_point()
            print("d", self.spawn_flag, self.d, self.count)

        elif col:
            self.count = 0
            self.spawn_x = self.point[self.sub_goal_count][0]
            self.spawn_y = self.point[self.sub_goal_count][1]
            self.spawn_h = self.point[self.sub_goal_count][2]
            self.spawn_flag = True
            # self.get_d(self.spawn_x, self.spawn_y, self.spawn_h)
            # print("col", self.spawn_flag)

        elif self.end_train:
            self.count = 0
            self.spawn_x = self.point[self.sub_goal_count][0]
            self.spawn_y = self.point[self.sub_goal_count][1]
            self.spawn_h = self.point[self.sub_goal_count][2]
            self.spawn_flag =True
            # self.get_d(self.spawn_x, self.spawn_y, self.spawn_h)
            # print("end_train", self.spawn_flag)
        else:
            self.spawn_flag = False

        return self.spawn_x, self.spawn_y, self.spawn_h, self.spawn_flag, self.count

    def get_action(self, state):
        action = []
        if np.random.rand() <= 0.01:
            action.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
            # self.random_prob = self.random_prob * self.random_dicount
            return action
        else:
            state = np.reshape(state, (-1, 4))
            self.prediction = self.model.predict(x=state)
            return self.prediction

    def read_data(self, ep):
        self.episode = ep

        train_data_file = open(self.train_data_path + 'dddqn' + str(self.episode) + '.txt', 'r')
        train_data_lines = train_data_file.readlines()
        train_data_length = len(train_data_lines)
        self.train_data_length = train_data_length

        self.state = np.zeros((self.train_data_length, self.input_shape))
        self.next_state = np.zeros((self.train_data_length, self.input_shape))
        self.action = np.zeros((self.train_data_length, 4))
        self.re = np.zeros((self.train_data_length, 1))
        self.ac = np.zeros((self.train_data_length, 1))

        for i in range(train_data_length):
            data = train_data_lines[i].split('\t')
            self.state[i][0] = float(data[0])
            self.state[i][1] = float(data[1])
            self.state[i][2] = float(data[2])
            self.state[i][3] = float(data[3])

            self.action[i][0] = float(data[4])
            self.action[i][1] = float(data[5])
            self.action[i][2] = float(data[6])
            self.action[i][3] = float(data[7])

            self.re[i][0] = float(data[8])
            self.ac[i][0] = float(data[9])

            self.next_state[i][0] = float(data[10])
            self.next_state[i][1] = float(data[11])
            self.next_state[i][2] = float(data[12])
            self.next_state[i][3] = float(data[13])

    def model_train(self, ep):
        self.read_data(ep)

        # start = time.clock()
        q_tar = self.target_model.predict(x=self.next_state, batch_size=self.batch_size)
        y_train = self.model.predict(x=self.state, batch_size=self.batch_size)

        # print("length", len(action))
        for i in range(self.batch_size):

            # ratio_a = self.action[i][0] / total
            # ratio_s = abs(self.action[i][1]) / total
            # ratio_b = self.action[i][2] / total
            # q_total = math.sqrt(math.pow(q_tar[i][0], 2) + math.pow(q_tar[i][1], 2) + math.pow(q_tar[i][2], 2))
            # # print("total", total, "ratio ", self.re[i], y_train[i])
            # y_train[i][0] = ratio_a * (self.re[i] + self.discount_factor * q_total)
            # y_train[i][1] = ratio_s * (self.re[i] + self.discount_factor * q_total)
            # y_train[i][2] = ratio_b * (self.re[i] + self.discount_factor * q_total)
            # print(y_train[i], q_tar[i], self.re[i])
            y_train[i][0] = q_tar[i][0] + self.ac[i]
            if q_tar[i][1] > q_tar[i][2]:
                y_train[i][1] = q_tar[i][1] + self.re[i]
            else:
                y_train[i][2] = q_tar[i][2] + self.re[i]
            y_train[i][3] = q_tar[i][3]


        self.model.fit(self.state, y_train, epochs=self.epochs, batch_size=self.batch_size)
        # end = time.clock()
        # print("training time : {} ".format((end - start)))
        self.end_train = True


    def build_model(self):
        inupt = Input(shape=(4, ))
        x = Dense(128, activation='relu')(inupt)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        value = Dense(4, activation='sigmoid')(x)
        # ac = Dense(1, activation='sigmoid')(x)
        # st = Dense(1, activation='tanh')(x)
        # br = Dense(1, activation='sigmoid')(x)

        # q = concatenate([ac, st, br])
        model = Model(inupt, value)
        # model = Model(inupt, value)
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model

    def soft_weight_update(self):
        train_network_weight = self.model.get_weights()
        target_network_weight = self.target_model.get_weights()
        for i in range(len(train_network_weight)):
            target_network_weight[i] = (self.Tau * train_network_weight[i]) + ((1-self.Tau) * target_network_weight[i])
        self.target_model.set_weights(target_network_weight)

