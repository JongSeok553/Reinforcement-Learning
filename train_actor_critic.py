from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, BatchNormalization, Flatten, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import random
import math
from collections import deque
_EPSILON = K.epsilon()

class A2C:
    def __init__(self):
        self.policy = self.build_policy()
        self.critic = self.build_critic()
        self.actor = self.build_actor()
        self.batch_size = 500
        self.memory = deque(maxlen=5000)
        self.learning_rate = 0.001
        self.input_shape = 6
        self.output_shape = 7
        self.img_row = 84
        self.img_col = 84
        self.epochs = 4
        self.discount_factor = 0.99

    def replaymemory(self, image, state, action, reward, next_image, nextstate):
        self.memory.append((image, state, action, reward, next_image, nextstate))

    def get_action(self, image, state):
        if random.randint(0, 100) < 1:
            return random.randint(0, 6)
        else:
            image = np.reshape(image, [1, 84, 84, 1])
            state = state
            policy = self.actor.predict([image, state], batch_size=1).flatten()
            return np.random.choice(self.output_shape, 1, p=policy)[0]

    @staticmethod
    def build_policy():
        image = Input(shape=(84, 84, 1))
        x = Conv2D(64, (6, 6), strides=(3, 3), activation='relu')(image)
        x = BatchNormalization()(x)
        x = Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        image_dense = Dense(16, activation='relu')(x)

        state = Input(shape=(6,))

        state_vector = Dense(32, activation='relu')(state)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(16, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(8, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        action = concatenate([image_dense, state_vector])

        action = Dense(7, activation='softmax')(action)

        policy_model = Model(inputs=[image, state], outputs=action,name='policy')
        policy_model.summary()

        return policy_model

    @staticmethod
    def build_actor():
        image = Input(shape=(84, 84, 1))

        x = Conv2D(64, (6, 6), strides=(3, 3), activation='relu')(image)
        x = BatchNormalization()(x)
        x = Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        image_dense = Dense(16, activation='relu')(x)

        state = Input(shape=(6,))

        state_vector = Dense(32, activation='relu')(state)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(16, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(8, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        action = concatenate([image_dense, state_vector])

        action = Dense(8, activation='relu')(action)
        action = BatchNormalization()(action)

        action = Dense(7, activation='softmax')(action)

        def custom_loss(y_true, y_pred):
            y_pred = K.clip(y_pred, _EPSILON, 1 - _EPSILON)
            out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
            return K.mean(out, axis=-1)

        actor_model = Model(inputs=[image, state], outputs=action, name='actor')
        actor_model.compile(optimizer=Adam(lr=0.001), loss=custom_loss, metrics=['accuracy'])
        actor_model.summary()
        return actor_model

    @staticmethod
    def build_critic():
        image = Input(shape=(84, 84, 1))
        x = Conv2D(64, (6, 6), strides=(3, 3), activation='relu')(image)
        x = BatchNormalization()(x)
        x = Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        image_dense = Dense(16, activation='relu')(x)

        state = Input(shape=(6,))

        state_vector = Dense(32, activation='relu')(state)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(16, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        state_vector = Dense(8, activation='relu')(state_vector)
        state_vector = BatchNormalization()(state_vector)

        value = concatenate([image_dense, state_vector])

        value = Dense(8, activation='relu')(value)
        value = BatchNormalization()(value)

        value = Dense(1, activation='sigmoid')(value)

        critic_model = Model(inputs=[image, state], outputs=value, name='critic')
        critic_model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        critic_model.summary()
        return critic_model

    def model_train(self):
        for i in range(self.epochs):
            mini_batch = random.sample(self.memory, self.batch_size)
            image = np.zeros((self.batch_size, self.img_col, self.img_row, 1))
            next_image = np.zeros((self.batch_size, self.img_col, self.img_row, 1))
            state = np.zeros((self.batch_size, self.input_shape))
            next_state = np.zeros((self.batch_size, self.input_shape))
            reward = []
            action = np.zeros((self.batch_size, 1))
            delta = np.zeros((self.batch_size, self.output_shape))
            target = np.zeros((self.batch_size, 1))
            for i in range(self.batch_size):
                image[i] = np.float32(mini_batch[i][0]/255.0)
                state[i] = np.float32(mini_batch[i][1])
                action[i] = np.int8(mini_batch[i][2])
                reward.append(mini_batch[i][3])
                next_image[i] = np.float32(mini_batch[i][4] / 255.0)
                next_state[i] = np.float32(mini_batch[i][5])

            image = np.reshape(image, (-1, 84, 84, 1))
            state = np.reshape(state, (-1, 6))
            next_image = np.reshape(next_image, (-1, 84, 84, 1))
            value = self.critic.predict([image, state])
            next_value = self.critic.predict([next_image, next_state])
            for i in range(self.batch_size):
                action_index = int(action[i])
                target[i] = reward[i] + (self.discount_factor * next_value[i][0])
                delta[i][action_index] = target[i] - value[i]
            self.critic.fit([image, state], target, epochs=3)
            self.actor.fit([image, state], delta, epochs=3)


    # @staticmethod
    # def actor_train():
    #     with tf.GradientTape() as tape:
    #       predictions = model(images)
    #       loss = TD_error(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)

# model = MyModel()
#
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
#
# optimizer = tf.keras.optimizers.Adam()
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#
# # @tf.function
#
#
# @tf.function
# def test_step(images, labels):
#   predictions = model(images)
#   t_loss = loss_object(labels, predictions)
#
#   test_loss(t_loss)
#   test_accuracy(labels, predictions)
#
# EPOCHS = 5
#
# for epoch in range(EPOCHS):
#   for images, labels in train_ds:
#     train_step(images, labels)
#
#   for test_images, test_labels in test_ds:
#     test_step(test_images, test_labels)
#
#   template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
#   print (template.format(epoch+1,
#                          train_loss.result(),
#                          train_accuracy.result()*100,
#                          test_loss.result(),
#                          test_accuracy.result()*100))
