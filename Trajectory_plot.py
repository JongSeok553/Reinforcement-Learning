import numpy as np
import matplotlib.pyplot as plt

class plot_trajectory:
    def __init__(self):
        self.version = 10000
        self.version2 = 20000
        self.version3 = 30000
        self.version4 = 40000
        self.version5 = 50000
        self.version6 = 70000
        self.version7 = 150000

        self.trajectory_path = 'model_supervised/trajectory/'
        # self.trajectory_name = 'Test_trajectory' + str(self.version) + '.txt'
        # self.trajectory_file = open(self.trajectory_path + self.trajectory_name, 'r')
        #
        # self.trajectory_name2 = 'Test_trajectory' + str(self.version2) + '.txt'
        # self.trajectory_file2 = open(self.trajectory_path + self.trajectory_name2, 'r')
        #
        # self.lines = self.trajectory_file.readlines()
        # self.lines2 = self.trajectory_file2.readlines()
        #
        # self.trajectory_length = len(self.lines)
        # self.trajectory_length2 = len(self.lines2)


        # self.x = np.zeros((self.trajectory_length, 1))
        # self.y = np.zeros((self.trajectory_length, 1))

    def file_read(self):
        self.original = 'Original_trajectory.txt'
        self.trajectory_name = 'Test_trajectory' + str(self.version) + '.txt'
        self.trajectory_name2 = 'Test_trajectory' + str(self.version2) + '.txt'
        self.trajectory_name3 = 'Test_trajectory' + str(self.version3) + '.txt'
        self.trajectory_name4 = 'Test_trajectory' + str(self.version4) + '.txt'
        self.trajectory_name5 = 'Test_trajectory' + str(self.version5) + '.txt'
        self.trajectory_name6 = 'Test_trajectory' + str(self.version5) + '.txt'
        self.trajectory_name7 = 'Test_trajectory' + str(self.version5) + '.txt'

        self.original_file = open(self.trajectory_path + self.original, 'r')
        self.trajectory_file = open(self.trajectory_path + self.trajectory_name, 'r')
        self.trajectory_file2 = open(self.trajectory_path + self.trajectory_name2, 'r')
        self.trajectory_file3 = open(self.trajectory_path + self.trajectory_name3, 'r')
        self.trajectory_file4 = open(self.trajectory_path + self.trajectory_name4, 'r')
        self.trajectory_file5 = open(self.trajectory_path + self.trajectory_name5, 'r')
        self.trajectory_file6 = open(self.trajectory_path + self.trajectory_name6, 'r')
        self.trajectory_file7 = open(self.trajectory_path + self.trajectory_name7, 'r')



        self.original_lines = self.original_file.readlines()
        self.lines = self.trajectory_file.readlines()
        self.lines2 = self.trajectory_file2.readlines()
        self.lines3 = self.trajectory_file3.readlines()
        self.lines4 = self.trajectory_file4.readlines()
        self.lines5 = self.trajectory_file5.readlines()
        self.lines6 = self.trajectory_file6.readlines()
        self.lines7 = self.trajectory_file7.readlines()



        self.original_length = len(self.original_lines)
        self.trajectory_length = len(self.lines)
        self.trajectory_length2 = len(self.lines2)
        self.trajectory_length3 = len(self.lines3)
        self.trajectory_length4 = len(self.lines4)
        self.trajectory_length5 = len(self.lines5)
        self.trajectory_length6 = len(self.lines6)
        self.trajectory_length7 = len(self.lines7)


        self.ox = np.zeros((self.original_length, 1))
        self.oy = np.zeros((self.original_length, 1))

        self.x = np.zeros((self.trajectory_length, 1))
        self.y = np.zeros((self.trajectory_length, 1))

        self.x2 = np.zeros((self.trajectory_length2, 1))
        self.y2 = np.zeros((self.trajectory_length2, 1))

        self.x3 = np.zeros((self.trajectory_length3, 1))
        self.y3 = np.zeros((self.trajectory_length3, 1))

        self.x4 = np.zeros((self.trajectory_length4, 1))
        self.y4 = np.zeros((self.trajectory_length4, 1))

        self.x5 = np.zeros((self.trajectory_length5, 1))
        self.y5 = np.zeros((self.trajectory_length5, 1))

        self.x6 = np.zeros((self.trajectory_length6, 1))
        self.y6 = np.zeros((self.trajectory_length6, 1))

        self.x7 = np.zeros((self.trajectory_length7, 1))
        self.y7 = np.zeros((self.trajectory_length7, 1))

    def plot(self):
        for i in range(self.trajectory_length):
            data = self.lines[i].split('\t')
            self.x[i][0] = float(data[0])
            self.y[i][0] = float(data[1])
        for i in range(self.trajectory_length2):
            data = self.lines2[i].split('\t')
            self.x2[i][0] = float(data[0])
            self.y2[i][0] = float(data[1])

        for i in range(self.trajectory_length3):
            data = self.lines3[i].split('\t')
            self.x3[i][0] = float(data[0])
            self.y3[i][0] = float(data[1])

        for i in range(self.trajectory_length4):
            data = self.lines4[i].split('\t')
            self.x4[i][0] = float(data[0])
            self.y4[i][0] = float(data[1])

        for i in range(self.trajectory_length5):
            data = self.lines5[i].split('\t')
            self.x5[i][0] = float(data[0])
            self.y5[i][0] = float(data[1])

        for i in range(self.trajectory_length6):
            data = self.lines6[i].split('\t')
            self.x6[i][0] = float(data[0])
            self.y6[i][0] = float(data[1])

        for i in range(self.trajectory_length7):
            data = self.lines7[i].split('\t')
            self.x7[i][0] = float(data[0])
            self.y7[i][0] = float(data[1])

        for i in range(self.original_length):
            data = self.original_lines[i].split('\t')
            self.ox[i][0] = float(data[0])
            self.oy[i][0] = float(data[1])

        plt.figure()
        plt.plot(self.oy, self.ox, label='Original')
        plt.legend(loc='lower left')
        plt.plot(self.y, self.x, label='epoch 10000')
        plt.legend(loc='lower left')
        plt.plot(self.y2, self.x2, label='epoch 20000')
        plt.legend(loc='lower left')
        plt.plot(self.y3, self.x3, label='epoch 30000')
        plt.legend(loc='lower left')
        plt.plot(self.y4, self.x4, label='epoch 40000')
        plt.legend(loc='lower left')
        plt.plot(self.y5, self.x5, label='epoch 50000')
        plt.legend(loc='lower left')
        plt.plot(self.y6, self.x6, label='epoch 70000')
        plt.legend(loc='lower left')
        plt.plot(self.y7, self.x7, label='epoch 150000')
        plt.legend(loc='lower left')
        plt.show()


if __name__ == '__main__':
    plot = plot_trajectory()
    plot.file_read()
    plot.plot()