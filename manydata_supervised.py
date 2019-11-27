from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import numpy as np
import matplotlib.pyplot as plt

class Train:
    def __init__(self):
        self.data_file_path = 'model_supervised/4action/'
        self.data_file_name = '4action_xyhd_all'
        self.f = open(self.data_file_path + self.data_file_name + '.txt', 'r')
        self.lines = self.f.readlines()
        self.data_length = len(self.lines)
        self.learning_rate = 0.001
        self.input_shape = 4
        self.output_shape = 4
        self.X_train = np.zeros((self.data_length, self.input_shape))
        self.Y_train = np.zeros((self.data_length, self.output_shape))
        self.accel = []
        self.steer = []
        self.brake = []
        self.model = self.build_model()
        self.hist = 0
        self.epochs = 2000
        self.batch_size = 500
        self.model_name = str(self.batch_size) + '_batch' + '_data_' + str(self.data_file_name) + '_Test' + str(self.epochs)

    def build_model(self):
        inupt = Input(shape=(4,))
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


    def train_model(self):
        for i in range(self.data_length):
            print(self.data_length)
            data = self.lines[i].split('\t')
            self.X_train[i][0] = float(data[0])
            self.X_train[i][1] = float(data[1])
            self.X_train[i][2] = float(data[2])
            self.X_train[i][3] = float(data[3])

            self.Y_train[i][0] = float(data[4])
            self.Y_train[i][1] = float(data[5])
            self.Y_train[i][2] = float(data[6])
            self.Y_train[i][3] = float(data[7])

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
        plt.savefig('model_supervised/plot/' + self.model_name + '.png')
        plt.show()
        print(self.model_name + '.png' + ' save plot file')


if __name__ == '__main__':
    train = Train()
    train.train_model()
    model_name = train.model_name
    train.model.save('model_supervised/train_data/' +str(2000)+'4action_xyhd_all' + '.h5')
    print("4action_xyhd_all" + '.h5' + " save weight file")
    train.result_plot()
