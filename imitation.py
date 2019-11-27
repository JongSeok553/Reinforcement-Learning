import numpy as np
import tensorflow as tf
import time


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
        self.make_point()
        self.x = 0
        self.y = 0


    def read_data(self, ep):

        test_data_file = open(self.train_data_path + 'dddqn' + str(self.episode) + '.txt', 'r')
        original_data_file = open(self.train_data_path + 'dddqn' + str(self.episode) + '.txt', 'r')

        test_data_lines = test_data_file.readlines()
        original_data_lines = original_data_file.readlines()

        test_data_length = len(test_data_lines)
        original_data_length = len(original_data_lines)

        self.x = np.zeros((test_data_length, 4))
        self.y = np.zeros((original_data_length, 4))

        for i in range(original_data_length):
            test_data = test_data_lines[i].split('\t')
            original_data = original_data_lines[i].split('\t')
            self.x[i][0] = float(test_data[0])
            self.x[i][1] = float(test_data[1])
            self.x[i][2] = float(test_data[2])
            self.x[i][3] = float(test_data[3])

            self.y[i][0] = float(original_data[0])
            self.y[i][1] = float(original_data[1])
            self.y[i][2] = float(original_data[2])
            self.y[i][3] = float(original_data[3])




        X_data = tf.placeholder(tf.float32)
        Y_data = tf.placeholder(tf.float32)

        W1 = tf.Variable(tf.random_uniform([4, 128], -1., 1.))
        W2 = tf.Variable(tf.random_uniform([128, 64], -1., 1.))
        W3 = tf.Variable(tf.random_uniform([64, 32], -1., 1.))
        W4 = tf.Variable(tf.random_uniform([32, 16], -1., 1.))
        W5 = tf.Variable(tf.random_uniform([16, 4], -1., 1.))

        L1 = tf.nn.relu(tf.matmul(X_data, W1))
        L2 = tf.nn.relu(tf.matmul(L1, W2))
        L3 = tf.nn.relu(tf.matmul(L2, W3))
        L4 = tf.nn.relu(tf.matmul(L3, W4))
        model = tf.nn.sigmoid(tf.matmul(L4, W5))

        cost = tf.reduce_mean(tf.square(X_data - Y_data))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in range(100):
            sess.run(train, feed_dict={X_data, Y_data})

            if (step + 1) % 10 == 0:
                print(step + 1, sess.run(cost, feed_dict={X_data, Y_data}))




