import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import GRU, LSTM, Reshape, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from emg_load_data import load_data
"""
(X_train, y_train), (X_test, y_test) = load_data()
"""
tf.random.set_seed(0)
np.random.seed(0)

# mnist dataset 의 간단한 데이터 전처리
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

leaky_relu = tf.nn.leaky_relu()

input_shape = Input(shape=[X_train.shape[1], X_train.shape[2], X_train.shape[3]])
lstm_input = Input(shape=[X_train.shape[1] * X_train.shape[2], X_train.shape[3]])
def cnn_2d():
    con2d = (Conv2D(32, kernel_size=(2, 2), padding='same', activation=leaky_relu))(input_shape)
    bn = (BatchNormalization())(con2d)
    max_polling2 = (MaxPooling2D((2, 2))(bn))
    con2d = (Conv2D(16, kernel_size=(2, 2), padding='same', activation=leaky_relu))(max_polling2)
    max_polling2 = (MaxPooling2D((2, 2))(con2d))
    flatten = (Flatten())(max_polling2)
    dense = (Dense(50, activation=leaky_relu))(flatten)
    dense = (BatchNormalization())(dense)
    dense = (Dense(20, activation=leaky_relu))(dense)
    return dense

def con_2d_2():
    con2d = (Conv2D(32, kernel_size=(2, 2), padding='same', activation=leaky_relu))(input_shape)
    bn = (BatchNormalization())(con2d)
    max_polling2 = (MaxPooling2D((2, 2))(bn))
    con2d = (Conv2D(16, kernel_size=(2, 2), padding='same', activation=leaky_relu))(max_polling2)
    max_polling2 = (MaxPooling2D((2, 2))(con2d))
    flatten = (Flatten())(max_polling2)
    dense = (Dense(50, activation=leaky_relu))(flatten)
    dense = (BatchNormalization())(dense)
    dense = (Dense(20, activation=leaky_relu))(dense)
    return dense


def lstm_modeling():
    reshape = Reshape(target_shape=(X_train.shape[1] * X_train.shape[2], X_train.shape[3]))(input_shape)
    lstm = (LSTM(10, return_sequences=False))(reshape)
    dense = Dense(20, activation='leaky_relu')(lstm)

    model_concat = tf.keras.layers.concatenate([cnn_2d(), con_2d_2(), dense], axis=1)
    dense = (Dense(10, activation='softmax'))(model_concat)

    k_model = tf.keras.models.Model(input_shape, dense)
    k_model.summary()
    tf.keras.utils.plot_model(k_model, 'test.png')
    adam = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam,
                    metrics=['acc'])

    k_model.fit(X_train, y_train, epochs=300, batch_size=128, validation_data=(X_test, y_test),
                verbose=1, callbacks=[callback])
    print(k_model.evaluate(X_test, y_test, batch_size=128, verbose=1))

    xhat_idx = np.random.choice(X_test.shape[0], 60000)
    xhat = X_test[xhat_idx]
    yhat = k_model.predict(xhat)

    result = 0
    loss = 0
    for i in range(60000):
        print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
        if str(np.argmax(X_test[xhat_idx[i]])) == str(yhat[i]):
            result += 1
        elif str(np.argmax(X_test[xhat_idx[i]])) != str(yhat[i]):
            loss += 1
        else:
            print("errors")
    print("result > " + str(result) + " loss > ", str(loss))


lstm_modeling()
