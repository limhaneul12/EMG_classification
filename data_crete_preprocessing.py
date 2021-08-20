import os
import pandas as pd

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection

def crate_image_data():
    w = 256
    h = 256
    folder = './img_direct/'
    categorical = ['Box_holding', 'Eating', 'Erasing-writing', 'Lifting_up_and_down',
                   'Light_bulb', 'Open-close-door', 'Reaching_out_and_retracting', 'Smart_phone_check']
    num_classes = len(categorical)
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

"""
경로를 자기 directory file 위치에 맞춰서 경로를 설정해줄것 
ex) /home/lmsky/PycharmProjects/torch_practice/data_time/Box_holding.csv 로 가고싶을때 
main_root = /home/lmsky/PycharmProjects/torch_practice/
file = data_time/
os.chdir(file)  # file(data_time)이라는 directory 안쪽으로 접근  
path = f'{main_root}{file}' 
"""
def image_visualization():
    # 최상단 위치
    main_root = '/home/lmsky/PycharmProjects/torch_practice/'
    # 하위 위치
    file = 'data_preprocessing/210814_emg_training_session/'
    os.chdir(file)
    # 현재 디렉토리 파일 list 형태로 보여줌
    location = os.getcwd()
    # 현재 디렉토리 파일 하위 폴더 위치 보여줌
    root = os.listdir(location)
    root.sort()
    for i in root:
        # 파일 루트 설정
        path_folder = f'{main_root}{file}{i}/1st/'
        # 이미지 저장할 루트 설정
        path_folder2 = f'{main_root}img_direct/{i}/'
        #  path_folder file list 형태로 보여줌
        root_data = os.listdir(path_folder)
        root_data.sort() # 정렬
        for j in root_data:
            a = pd.read_csv(path_folder+j)
            col = a.columns
            ax = plt.gca()

            ax.set_facecolor('white')
            plt.plot(a[col[0]], a[col[1]], c='black')
            # 경고 무시해도됨
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.axis('off')
            plt.savefig(f'{path_folder2}/{i}__{j}.png', facecolor='white')
            plt.show()
