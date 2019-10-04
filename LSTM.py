from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

class lstm_train:
    def __init__(self):
        self.f = open('data.txt', 'r')
        self.lines = self.f.readlines()
        self.data_length = len(self.lines)

        self.data_dim = 2
        self.input_time_step = 5
        self.output_time_stpe = 3
        self.output_shape = 3
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 1000

        self.model = self.build_model()
        self.X_train, self.Y_train = self.make_data()
        # np.reshape(self.X_train, (len(self.X_train), self.time_step, 1))
        # np.reshape(self.Y_train, (len(self.Y_train), self.output_shape, 1))
        print(self.X_train.shape)
        print(self.Y_train.shape)


    def make_data(self):
        x_train = np.zeros((self.data_length, self.input_time_step, self.data_dim))
        y_train = np.zeros((self.data_length, self.output_time_stpe, self.output_shape))

        for i in range(self.data_length):
            data = self.lines[i].split('\t')
            for j in range(self.input_time_step):
                x_train[i][j][0] = float(data[0])
                x_train[i][j][1] = float(data[1])
            print(x_train.shape)

        for i in range(self.data_length):
            data = self.lines[i].split('\t')
            for j in range(self.output_time_stpe):
                y_train[i][j][0] = float(data[2])
                y_train[i][j][1] = float(data[3])
                y_train[i][j][2] = float(data[4])
            print(y_train.shape)
        return x_train, y_train

    def build_model(self):
        model = Sequential()    # stateful 은 이전 배치의에서 학습한 상태가 다음배치에 사용됨 # return_sequence = False로하면 가능
        model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=
            (self.batch_size, self.input_time_step, self.data_dim)))
        model.add(LSTM(32,return_sequences=True, stateful=True))
        model.add(LSTM(32, return_sequences=True, stateful=True))
        model.add(Dense(self.output_shape, activation='tanh'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def train_model(self):
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size)

if __name__ == '__main__':
    train = lstm_train()
    train.train_model()
    train.model.save('model_lstm/' + 'LSTM_Test' + str(train.epochs) + '.h5')
    print('LSTM_Test' + str(train.epochs) + '.h5' + " save weight file")
