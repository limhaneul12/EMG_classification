import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling1D, Conv1D
from tensorflow.keras.layers import GRU, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import StandardScaler

epoch = 50
batch_size = 30
num_classes = 11
data = 'test1111.csv'

# 64 128 256
def build_model(data_len):
    input_shape = [data_len, data]
    input_shape = tf.keras.Input(shape=[data_len])
    model = (Conv1D(32, kernel_size=3, padding='same', activation='relu'))(input_shape)
    model = (MaxPooling1D(3))(model)
    model = (Dropout(0.25))(model)
    model = (Conv1D(64, kernel_size=3, padding='same', activation='relu'))(model)
    model = (MaxPooling1D(3))(model)
    model = (Dropout(0.3))(model)
    model = (Conv1D(128, kernel_size=3, padding='same', activation='relu'))(model)
    model = (MaxPooling1D(3))(model)
    model = (LSTM(50))(model)
    model = (Flatten())(model)
    model = (Dense(128, activation='relu'))(model)
    model = (Dropout(0.25))(model)
    model = (Dense(16, activation='relu'))(model)
    model = (Dense(40, activation='softmax'))(model)

    k_model = tf.keras.models.Model(input_shape, model)
    k_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return k_model


def cnn_2d():
    con2d = (Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
    max_polling2 = (MaxPooling2D((3, 3))(con2d))
    con2d = (Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))(max_polling2)
    max_polling2 = (MaxPooling2D((3, 3))(con2d))
    con2d = (Conv2D(16, kernel_size=(2, 2), padding='same', activation='relu'))(max_polling2)
    max_polling2 = (MaxPooling2D((3, 3))(con2d))

    return max_polling2

def con_2d_2():
    con2d = (Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
    max_polling2 = (MaxPooling2D((3, 3))(con2d))
    con2d = (Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))(max_polling2)
    max_polling2 = (MaxPooling2D((3, 3))(con2d))
    con2d = (Conv2D(16, kernel_size=(2, 2), padding='same', activation='relu'))(max_polling2)
    max_polling2 = (MaxPooling2D((3, 3))(con2d))

    return max_polling2


def lstm_modeling():
    lstm = (LSTM(10, return_sequences=True))(inpu)
    lstm = (LSTM(10, return_sequences=True))(lstm)
    lstm = (LSTM(10, return_sequences=True))(lstm)
    lstm = (LSTM(10, return_sequences=True))(lstm)

    model_concat = tf.keras.layers.concatenate(cnn_2d(), lstm)

    flatten = (Flatten())(model_concat)
    dense = (Dense(128, activation='relu'))(flatten)
    dense = (Dense(128, activation='relu'))(dense)
    dense = (Dropout(0.25))(dense)
    dense = (Dense(7, activation='softmax'))(dense)

    k_model = tf.keras.models.Model(input_shape, dense)
    k_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


