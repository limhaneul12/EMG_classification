import os
import glob

import cv2
import numpy as np
import sklearn.model_selection

folder = './img_direct/'
categorical = ['Box_holding', 'Eating', 'Erasing-writing', 'Lifting_up_and_down',
               'Light_bulb', 'Open-close-door', 'Reaching_out_and_retracting', 'Smart_phone_check']
num_classes = len(categorical)

w = 256
h = 256

x = []
y = []

for index, categorical in enumerate(categorical):
    label = [0 for i in range(num_classes)]
    label[index] = 1
    dir_ = f'{folder+categorical}/'

    for top, dir1, f in os.walk(dir_):
        for filename in f:
            print(dir_+filename)
            img = cv2.imread(dir_ + filename)
            img = cv2.resize(img, None, fx=w/img.shape[1], fy=h/img.shape[0])
            x.append(img/256)
            y.append(label)

x = np.array(x)
y = np.array(y)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)
np.savez('test_image_data.npz', x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)
