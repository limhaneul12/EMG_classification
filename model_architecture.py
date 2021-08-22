import datetime
import os
import random
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import LSTM, Reshape, Input, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

from confusion_shared import print_confusion_matrix_v2
from confusion_shared import logw
"""
from emg_load_data import load_data
# signal dataset
(X_train, y_train), (X_test, y_test) = load_data()
"""
# 난수 고정 컴퓨터는 추상화 능력이 없어 값을 임의적으로 바꿀 수 없으니 값을 고정
tf.random.set_seed(0)
np.random.seed(0)

# mnist dataset 의 간단한 데이터 전처리
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalization deep learning early training speed calculation
# 0 ~ 1 scale, channel data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용
print(f"Shape checking X_train image: {X_train.shape}")
print(f"Shape checking X_test image: {X_test.shape}")

# 라벨링 mnist 는 0~9 총 10개
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

epochs = 100
batch_size = 128
learning_rate = 1e-3
num_classes = 10

def image_show(n, image_size=256):
    image = X_train[n]
    image_reshaped = image.reshape(image_size, image_size)
    label = X_train[n]
    plt.figure(figsize=(4, 4))
    plt.title("sample of " + str(label))
    plt.imshow(image_reshaped, cmap="gray")
    plt.show()


# input data shape
input_shape = Input(shape=[X_train.shape[1], X_train.shape[2], X_train.shape[3]])

activate_leaky_relu = tf.keras.layers.LeakyReLU()


# 범용 활성화함수
# activation 입니다 False 라고 한 이유는 각 activation function parameter False 는 사용안함 True 하면 사용
def activation_optional(input_size, alpha=0.3, leaky_relu=False, relu=False):
    norm = tf.keras.layers.BatchNormalization()(input_size)
    if leaky_relu or alpha:  # leaky_relu 나 alpha 값이 일치하면 실행
        activation = tf.keras.layers.LeakyReLU(alpha=alpha)(norm)
    else:
        if relu:  # 그렇지 않으면 다음과 같이 실행
            activation = tf.keras.layers.ReLU()(norm)
        else:
            activation = tf.keras.layers.ELU()(norm)

    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(activation)


def cnn_model_arch1():
    con2d = (Conv2D(64, kernel_size=(3, 3), padding='same', activation=activate_leaky_relu))(input_shape)
    bn = (activation_optional(con2d))
    con2d = (Conv2D(64, kernel_size=(3, 3), padding='same', activation=activate_leaky_relu))(bn)
    bn = (activation_optional(con2d))
    con2d = (Conv2D(32, kernel_size=(3, 3), padding='same', activation=activate_leaky_relu))(bn)
    bn_polling = (activation_optional(con2d))
    drop_out = (Dropout(0.25))(bn_polling)

    flatten = (Flatten())(drop_out)
    dense = (Dense(512, activation=activate_leaky_relu))(flatten)
    dense = (BatchNormalization())(dense)
    drop_out = (Dropout(0.25))(dense)
    dense = (Dense(1024, activation=activate_leaky_relu))(drop_out)
    bn = (BatchNormalization())(dense)
    drop_out = (Dropout(0.5))(bn)

    return drop_out


def cnn_model_arch2():
    con2d = (Conv2D(32, kernel_size=(3, 3), padding='same', activation=activate_leaky_relu))(input_shape)
    bn = (activation_optional(con2d))
    con2d = (Conv2D(32, kernel_size=(3, 3), padding='same', activation=activate_leaky_relu))(bn)
    bn = (activation_optional(con2d))
    con2d = (Conv2D(16, kernel_size=(3, 3), padding='same', activation=activate_leaky_relu))(bn)
    bn_polling = (activation_optional(con2d))
    drop_out = (Dropout(0.25))(bn_polling)

    flatten = (Flatten())(drop_out)
    dense = (Dense(512, activation=activate_leaky_relu))(flatten)
    dense = (BatchNormalization())(dense)
    drop_out = (Dropout(0.25))(dense)
    dense = (Dense(1024, activation=activate_leaky_relu))(drop_out)
    bn2 = (BatchNormalization())(dense)
    drop_out = (Dropout(0.5))(bn2)

    return drop_out


def lstm_modeling():
    reshape = Reshape(target_shape=(X_train.shape[1], X_train.shape[2]))(input_shape)
    lstm = (LSTM(10, return_sequences=True))(reshape)
    lstm = (LSTM(10, return_sequences=True))(lstm)
    lstm = (LSTM(10, return_sequences=True))(lstm)
    lstm = (LSTM(10, return_sequences=True))(lstm)
    flatten = (Flatten())(lstm)

    return flatten


# cnn1 cnn 2 lstm concatnate
def data_concatnate():
    model_concat = tf.keras.layers.concatenate([cnn_model_arch1(), cnn_model_arch2(), lstm_modeling()])
    finally_dense = (Dense(num_classes, activation='softmax'))(model_concat)
    k_model = tf.keras.models.Model(input_shape, finally_dense)
    tf.keras.utils.plot_model(k_model, 'modeling_data_64_32_batch_drop.png', show_shapes=True)
    k_model.summary()
    return k_model


def model_fitting():
    k_model = data_concatnate()
    # optimizer
    adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)
    # 과적합 방지
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    # modeling fitting
    history = k_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                          validation_data=(X_test, y_test),
                          verbose=1, callbacks=[callback])
    print(k_model.save('test_cnn_32_batch_dropout.h5'))
    score, acc = k_model.evaluate(X_test, y_test, batch_size=128, verbose=1)

    # model training graph visualization
    def visualization():
        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()

        loss_ax.plot(history.history['loss'], 'y', label='train loss')
        loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(history.history['acc'], 'b', label='train acc')
        acc_ax.plot(history.history['val_acc'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.show()


    # 건들지 마세요 model image classification test
    def prediction_data():
        prediction_result = k_model.predict(X_test)
        prediction_labels = np.argmax(prediction_result, axis=-1)
        test_label = np.argmax(y_test, axis=-1)

        fnpath = 'result/'
        try:
            os.mkdir(fnpath)
        except OSError as e:
            print(f'An error has occurred. Continuing anyway: {e}')

        filename = f'{os.getcwd()}/classification_output.txt'
        file = open(filename, 'a+')
        logw(file, 'result -> {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

        confusion = print_confusion_matrix_v2(prediction_result, y_test)
        logw(file, f'Model Test loss -> {score} , Model Test accuracy -> {acc}')
        logw(file, f'Confusion Matrix -> \n' + np.array2string(confusion))
        wrong_result = []
        for n in range(0, len(test_label)):
            if prediction_labels[n] == test_label[n]:
                wrong_result.append(n)

        sample = random.choices(population=wrong_result, k=16)
        count = 0
        nrows = ncols = 4
        plt.figure(figsize=(12, 8))
        for n in sample:
            count += 1
            plt.subplot(nrows, ncols, count)
            plt.imshow(X_test[n].reshape(28, 28), cmap="Greys", interpolation="nearest")
            tmp = "Label:" + str(test_label[n]) + ", Prediction:" + str(prediction_labels[n])
            plt.title(tmp)

        plt.tight_layout()
        plt.show()

    def classes_predict():
        x_hat_idx = np.random.choice(X_train.shape[0], 60000)
        x_hat = X_train[x_hat_idx]

        prediction_result = k_model.predict(x_hat)
        prediction_labels = np.argmax(prediction_result, axis=-1)
        test_label = np.argmax(y_test, axis=-1)

        result = 0
        loss = 0
        for i in range(60000):
            if str(prediction_labels) == str(test_label):
                result += 1
            else:
                loss += 1

        print("result > {}".format(result))
        print("loss > {}".format(loss))

    visualization()
    prediction_data()
    classes_predict()


model_fitting()
