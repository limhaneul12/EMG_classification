import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# mnist dataset 의 간단한 데이터 전처리
(X_train, y_train), (X_test, y_test) = mnist.load_data()
k_model = load_model('test_cnn_64_batch_dropout.h5', custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU()})

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


prediction_result = k_model.predict(X_test)
prediction_labels = np.argmax(prediction_result, axis=-1)
test_label = np.argmax(y_test, axis=-1)


def classes_predict():
    result = 0
    loss = 0
    for n in range(0, len(test_label)):
        if prediction_labels[n] == test_label[n]:
            result += 1
        elif prediction_labels[n] != test_label[n]:
            loss += 1

    print(result, loss)


classes_predict()
