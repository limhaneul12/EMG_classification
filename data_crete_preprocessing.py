import os
import pandas as pd
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from PIL import Image
from scipy import signal

"""
일단 refactor 까지는 안해도 되는데 불편한 부분이 있을 수 있으므로 
여유가 된다면 class 화 하고 그렇지 않고 경로에 익숙해졋다고 하면 그냥 이상태로 ㄱㄱ  
"""

"""
데이터 경로를 꼭 맞춰주고 진행하세요 !!
segment_root 에 있는 1st 는 box_holding 에 있는 1st file 기준으로 맞춰준것이기에 꼭 경로 수정해주세요!


주의사항 
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
file 읽을때 띄어쓰기 로 된 file은 못읽어올 수 있음 
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


main_root = '/home/lmsky/PycharmProjects/torch_practice/'
file = data_preprocessing/210814_emg_training_session/
os.chdir(file) 이 부분은 file 의 자식 파일(하위 폴더가 무엇이 있는지) 찾는거고 

location = os.getcwd()  # 해당 폴더 위치를 반환함 
print(location)
output -> /home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session

location 안에 있는 file 를 찾아내서 리스트로 반홤함 
root = os.listdir(location)
root.sort() 
print(root) 
이렇게 파일(리스트형식) 획득 
['Lifting_up_and_down', 'Open-close-door', 'Reaching_out_and_retracting', 'Smart_phone_check', 
'Box_holding', 'Erasing-writing', 'Eating', 'Light_bulb']

이렇게 획득한 경로에 
segment_root = [f'{main_root}{file}{file_name}/1st/' for file_name in root] 
이렇게 써주게 되면 csv 파일 경로 획득할 수 있음 

스압 주의 segment_root output ->[' 
/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Box_holding/1st/', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Eating/1st/', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Erasing-writing/1st/', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Lifting_up_and_down/1st/', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Light_bulb/1st/', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Open-close-door/1st/', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Reaching_out_and_retracting/1st/', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Smart_phone_check/1st/' 
] 리스트 형식에 경로를 얻게됨 for문으로 해체하고  

169번째 줄 glob.glob 해서 *.csv 획득하면 다음과 같이 glob 값 나오게 되는데 [
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Box_holding/1st/EMGData_Back_0814.csv', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Box_holding/1st/EMGData_Biceps_0814.csv', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Box_holding/1st/EMGData_Front deltoids_0814.csv', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Box_holding/1st/EMGData_Pectoralis Major_0814.csv', 
'/home/lmsky/PycharmProjects/torch_practice/data_preprocessing/210814_emg_training_session/Box_holding/1st/EMGData_Triceps_0814.csv'
] for문으로 list를 풀어버리고 pd.read_csv 에 이 경로가 넣어져 csv 파일이 실행되는 원리 

"""
def csv_file_spilt():
    # 최상단 위치
    main_root = '/home/lmsky/PycharmProjects/torch_practice/'
    # 하위 위치
    file = 'data_preprocessing/210814_emg_training_session/'
    os.chdir(file)
    # 현재 디렉토리 파일 하위 폴더 위치 보여줌
    location = os.getcwd()
    print(location)
    # 현재 디렉토리 파일 list 형태로 보여줌
    root = os.listdir(location)
    print(root)
    root.sort()
    segment_root = [f'{main_root}{file}{file_name}/1st/' for file_name in root]
    print(segment_root)

    segment_root.sort()

    for loc in segment_root:
        data = glob.glob(loc + '*.csv')
        data.sort()
        print(data)
        for file in data:
            rowsize = 126
            # start looping through data writing it to a new file for each set
            number_lines = sum(1 for row in (open(file)))
            for i in range(1, number_lines, rowsize):
                df = pd.read_csv(file,
                                 nrows=rowsize,  # number of rows to read at each loop
                                 skiprows=i)  # skip rows that have been read

                out_csv = str(file)
                df.to_csv(f'{out_csv}',
                          index=False,
                          mode='a',  # append data to csv file
                          chunksize=rowsize)  # size of data to append for each loop


"""
이것도 위같은 방향으로 경로 설정  
거의 같은 경로여서 어렵진 않을꺼임 
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
        root_data.sort()  # 정렬
        for j in root_data:
            print(j)
            data = pd.read_csv(path_folder + j)
            fs = 1 / (data.time[1] - data.time[0])
            bh, ah = signal.butter(N=4, Wn=20 / (0.5 * fs), btype='highpass')
            bl, al = signal.butter(N=4, Wn=1 / (0.5 * fs), btype='lowpass')
            emg = np.fabs(signal.filtfilt(bh, ah, data.channel1))
            emg = signal.filtfilt(bl, al, emg)
            plt.figure(figsize=(20, 5))
            plt.plot(data['time'], emg)
            # 경고 무시해도됨
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.axis('off')
            # 데이터셋 마다 출력값이 달라서 y값 범위를 다 못채워 출력이 안되는 현상 발견함 돌려보면 앎 ㅇㅇ..
            # 데이터셋마다 y값을 바꿔야하는 상황
            plt.ylim([-0.5 * 10 ** -5, 6 * 10 ** -5])  # Y축의 범위: [ymin, ymax]
            plt.savefig(f'{path_folder2}/{i}__{j}.png', facecolor='black')
            plt.show()


def crate_image_data():
    # 그냥 임시로 해논거 바꿔야하는거 맞음 높이 너비
    w = 256
    h = 256
    folder = './img_direct/'
    """
    image dataset 만들때는 폴더 하나가 label 하나라고 보면 됨 
    이걸 보면 내 컴퓨터에는 Box_holding 이라는 폴더가 label 이다 라고 보면됨 
    그러면 총 8개의 label 이 있다 라고 보면 되는것!
    """
    # data labeling
    categorical = ['Box_holding', 'Eating', 'Erasing-writing', 'Lifting_up_and_down',
                   'Light_bulb', 'Open-close-door', 'Reaching_out_and_retracting', 'Smart_phone_check']
    num_classes = len(categorical)  # 총 갯수
    x = []
    y = []

    for index, categorical in enumerate(categorical):
        label = [0 for _ in range(num_classes)]
        label[index] = 1
        dir_ = f'{folder + categorical}/'

        for top, dir1, f in os.walk(dir_):
            for filename in f:
                print(dir_ + filename)
                img = Image.open(dir_ + filename).convert('L')
                img = np.array(img, 'uint8')
                print(img)
                """
                image 읽어서 resize 화 
                """
                img = cv2.resize(img, None, fx=w / img.shape[1], fy=h / img.shape[0])
                print(np.array(img).shape)
                x.append(img)  # 정규화 를 미리 해줫음 픽셀은 0~ 255 값인데 학습하기 위해서 0 ~ 1 사이 소수로 바꿈
                y.append(label)

    x = np.array(x)
    y = np.array(y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)
    np.savez('test_image_data1.npz', x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)
    print('done!')


"""
함수 사용하는 방법 
그냥 함수 이름 쓰면됨 
ex) 
csv_file_spilt() 요런식으로 

실행 순서
csv_file_spilt()
image_visualization()
create_image_data()
"""