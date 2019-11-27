from keras.models import Sequential, save_model, Model, load_model
from keras.layers import Dense, Lambda, Input, Add, Average, Subtract, average, BatchNormalization
from keras.optimizers import Adam
from collections import deque

import numpy as np
import math
import random

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
        self.sub_goal = []
        self.sub_goal_count = 0
        self.dest = 0
        self.pre_dest = 0

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
        self.last_count = 0
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.input_shape = 6
        self.output_shape = 3
        self.epochs = 30
        self.batch_size = 100
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.Tau = 0.99
        self.discount_Tau = 0.99
        self.train_start = False
        self.prediction =[]
        self.angle = 0
        self.d = 0
        self.end_train = False
        self.head_d = 0
        self.train_count = 0
        self.respawn_count = 0

        self.train_data_length = 0
        self.state = 0
        self.next_state = 0
        self.action = 0
        self.steer = 0
        self.re = 0

        self.gradient = 0
        self.reciprocal = 0
        self.episilon = 0.000001
        self.random_prob = 0.9
        self.random_dicount = 0.9

    def done(self):
        self.sub_goal_count = 0
        self.count = 0
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
        # print("next point", self.x2, self.cur_x, self.y2, self.cur_y, self.count, self.d)
        # print("next point", self.count, self.last_count)

    def make_point(self):
        for i in range(self.data_length):
            data = self.lines[i].split('\t')
            if i % 60 == 0:
                self.sub_goal.append((float(data[0]), float(data[1])))

            self.point.append((float(data[0]), float(data[1]), float(data[2])))
        print(len(self.sub_goal))
    def next(self, cur_x, cur_y):
        if self.count + 1 > self.data_length:
            self.done()
        else:

            d1 = math.sqrt(math.pow((self.x - cur_x), 2) + math.pow((self.y - cur_y), 2))
            d2 = math.sqrt(math.pow((self.x2 - cur_x), 2) + math.pow((self.y2 - cur_y), 2))
            p3_p2 = math.sqrt(math.pow((self.x3 - self.x2), 2) + math.pow((self.y3 - self.x2), 2))
            p3_p1 = math.sqrt(math.pow((self.x3 - self.x), 2) + math.pow((self.y3 - self.x), 2))
            # self.count += 1
            # self.next_point()
            # print("x1, x2,",self.x, self.x2, "y1, y2,",self.y, self.y2, "d1, d2, ",d1, d2, "p3_p2", p3_p2, "p3_p1", p3_p1)
            if d1 > d2:
                self.count += 1
                self.next_point()
            else:
                if p3_p2 >= p3_p1:
                    self.count += 1
                    self.next_point()

    def heading_d(self,cur_h):
        self.head_d = 0
        if self.point[self.sub_goal_count][2] > 0:
            if abs(self.point[self.sub_goal_count][2] - cur_h) > 180:
                self.head_d = -360 + abs(self.point[self.sub_goal_count][2] - cur_h)
                # print("set_h, cur_h, dis_h",self.point[self.sub_goal_count][2], cur_h, self.head_d)
            else:
                self.head_d = self.point[self.sub_goal_count][2] - cur_h
                # print("set_h, cur_h, dis_h", self.point[self.sub_goal_count][2], cur_h, self.head_d)
        else:
            if abs(self.point[self.sub_goal_count][2] - cur_h) > 180:
                self.head_d = 360 - abs(self.point[self.sub_goal_count][2] - cur_h)
                # print("set_h, cur_h, dis_h", self.point[self.sub_goal_count][2], cur_h, self.head_d)
            else:
                self.head_d = self.point[self.sub_goal_count][2] - cur_h
                # print("set_h, cur_h, dis_h", self.point[self.sub_goal_count][2], cur_h, self.head_d)

        return self.head_d

    def get_dest(self, curx, cury):
        sub_x = self.sub_goal[self.sub_goal_count][0]
        sub_y = self.sub_goal[self.sub_goal_count][1]

        reward = 0
        self.dest = math.sqrt((sub_x - curx) ** 2 + (sub_y - cury) ** 2)
        # print("dest", self.dest, self.sub_goal_count)
        if self.dest < 1.5:
            reward += 1
            self.sub_goal_count += 1

        if (self.pre_dest - self.dest) > 0.1:
            if self.d < 1:
                reward += 0.1
        else:
            reward += -0.2
        # print(reward)
        self.pre_dest = self.dest

        return self.dest, reward

    def get_d(self, cur_x, cur_y, cur_h):
        self.cur_x = cur_x
        self.cur_y = cur_y
        self.cur_h = cur_h
        b = self.y2 - self.y
        a = self.x2 - self.x
        c = (-b * self.x) + (a * self.y)
        self.gradient = a / (b + self.episilon)
        self.next(cur_x, cur_y)
        hd = self.heading_d(self.cur_h)
        self.dest, re = self.get_dest(self.cur_x, self.cur_y)
        if a == 0:
            self.d = 0  # abs(cur_y - self.y2)
        elif b == 0:
            self.d = 0  # abs(cur_x - self.x2)
        elif a == 0 and b == 0:
            self.d = 0
        else:
            self.d =abs(b * cur_x + (-a * cur_y) + c) / (
                        math.sqrt((math.pow(a, 2)) + (math.pow(b, 2))) + self.episilon)
        print("d ", self.d, "dest", self.dest, "count ", self.sub_goal_count)
        return self.d, self.dest, re, hd

    def replaymemory(self,state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        # print(action, reward)

    def respawn(self, col):
        # self.respawn_count +=1
        if abs(self.d) > 4:
            self.count = 0
            if self.sub_goal_count < 1:
                self.sub_goal_count = 1
            self.spawn_x = self.sub_goal[self.sub_goal_count-1][0]
            self.spawn_y = self.sub_goal[self.sub_goal_count-1][1]
            self.spawn_h = self.point[self.sub_goal_count-1][2]
            self.count = self.sub_goal_count * 61
            self.spawn_flag = True

        elif col:
            if self.sub_goal_count < 1:
                self.sub_goal_count = 2
            self.spawn_x = self.sub_goal[self.sub_goal_count-1][0]
            self.spawn_y = self.sub_goal[self.sub_goal_count-1][1]
            self.spawn_h = self.point[self.sub_goal_count-1][2]
            self.count = self.sub_goal_count * 61
            self.spawn_flag = True
            self.get_d(self.spawn_x, self.spawn_y, self.spawn_h)
            # print("col", self.spawn_flag)

        elif self.end_train:
            if self.sub_goal_count < 1:
                self.sub_goal_count = 2
            self.spawn_x = self.cur_x
            self.spawn_y = self.cur_y
            self.spawn_h = self.cur_h
            self.count = self.sub_goal_count * 61
            self.spawn_flag =True
            # self.get_d(self.spawn_x, self.spawn_y, self.spawn_h)
            # print("end_train", self.spawn_flag)
        else:
            self.spawn_flag = False

        return self.spawn_x, self.spawn_y, self.spawn_h, self.spawn_flag

    def get_action(self, state):
        if np.random.rand() <= 0.01:
            action = random.randint(0, 2)
            return action
        else:
            state = np.reshape(state, (-1, 6))
            self.prediction = self.model.predict(x=state)
            return np.argmax(self.prediction)

    # def read_data(self, ep):
    #     self.episode = ep
    #     train_data_file = open(self.train_data_path + 'dddqn' + str(self.episode) + '.txt', 'r')
    #     train_data_lines = train_data_file.readlines()
    #     train_data_length = len(train_data_lines)
    #     self.train_data_length = train_data_length
    #
    #     self.state = np.zeros((self.train_data_length, self.input_shape))
    #     self.next_state = np.zeros((self.train_data_length, self.input_shape))
    #     self.action = np.zeros((self.train_data_length, 1))
    #     self.re = np.zeros((self.train_data_length, 1))
    #
    #     for i in range(self.train_data_length):
    #         data = train_data_lines[i].split('\t')
    #         self.state[i][0] = float(data[0]) # velocity
    #         # self.state[i][1] = float(data[1])
    #         self.state[i][1] = float(data[2]) # heading
    #         self.state[i][2] = float(data[3]) # distance
    #         self.state[i][3] = float(data[4]) # heading distance
    #
    #         self.action[i][0] = float(data[5])
    #
    #         self.re[i][0] = float(data[6])
    #         #
    #         # # self.next_state[i][0] = float(data[7])
    #         # # self.next_state[i][1] = float(data[8])
    #         # self.next_state[i][0] = float(data[9])
    #         # # self.next_state[i][0] = float(data[10])
    #         # self.next_state[i][1] = float(data[11])

    def model_train(self, train_count):

        mini_batch = random.sample(self.memory, self.batch_size)
        self.state = [] #np.zeros((self.batch_size, self.input_shape))
        self.next_state = [] # np.zeros((self.batch_size, self.input_shape))
        self.action = []
        self.re = []

        for i in range(self.batch_size):
            self.state.append(mini_batch[i][0])
            self.action.append(mini_batch[i][1])
            self.re.append(mini_batch[i][2])
            self.next_state.append(mini_batch[i][3])
            # print(mini_batch[i])
        self.state = np.reshape(self.state, (-1, 6))
        self.next_state = np.reshape(self.next_state, (-1, 6))
        q_tar = self.target_model.predict(x=self.next_state)
        y_train = self.model.predict(x=self.state)
        # print(self.state)
        for i in range(self.batch_size):
            y_train[i][int(self.action[i])] = self.re[i] + self.discount_factor * (q_tar[i][self.action[i]])
        self.model.fit(self.state, y_train, epochs=self.epochs, batch_size=self.batch_size)
        self.end_train = True

    def build_model(self):
        inupt = Input(shape=(6, ))
        x = Dense(32, activation='relu')(inupt)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(inupt)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(8, activation='relu')(x)
        x = BatchNormalization()(x)
        value = Dense(3, activation='sigmoid')(x)

        model = Model(inupt, value)
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model

    def soft_weight_update(self):
        if self.Tau < 0.1:
            self.Tau = 0.1
        train_network_weight = self.model.get_weights()
        target_network_weight = self.target_model.get_weights()
        for i in range(len(train_network_weight)):
            target_network_weight[i] = (self.Tau * train_network_weight[i]) + ((1-self.Tau) * target_network_weight[i])
        self.target_model.set_weights(target_network_weight)
        self.Tau = self.Tau * self.discount_Tau
        print("episode ", self.episode," Tau ", self.Tau, "sub_point ", self.sub_goal_count, self.count)

