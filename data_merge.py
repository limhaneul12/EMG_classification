import glob
import os

import pandas as pd
import matplotlib.pyplot as plt

"""
Data is each other data checking data of shape (완료)
Data concatenate 하기 
"""
# 최상단 위치
main_root = '/home/lmsky/PycharmProjects/torch_practice/'
# 하위 위치
file = 'data_preprocessing/210814_emg_training_session/'
os.chdir(file)
# 현재 디렉토리 파일 list 형태로 보여줌
location = os.getcwd()
# 현재 디렉토리 파일 하위 폴더 위치 보여줌
root = os.listdir(location)
segment_root = [f'{main_root}{file}{file_name}/1st/' for file_name in root]
segment_root.sort()

def location():
    global segment_root
    emg_data = []
    for loc in segment_root:
        data = glob.glob(loc + '*.csv')
        data.sort()
        emg_data.append(data)
    return emg_data

def extraction():
    data = location()
    ttt = []
    for i in data:
        for j in i:
            ttt.sort()
            ttt.append(j)
    structure = [pd.read_csv(data_load) for data_load in ttt]
    data_concat = pd.concat(structure, axis=1).dropna(axis=0).drop()
    print(data_concat)
    data_concat.to_csv(f'{main_root}/test.csv', index_label=False, index=False)

def data_structure():
    data = location()
    for i in range(0, len(data)):
        concat_data = data[i]
        print(f'{i}번째 입니다 경로 -> {concat_data}')
        structure = [pd.read_csv(data_load) for data_load in concat_data]
        data_concat = pd.concat(structure, axis=1).dropna(axis=0)
        print(data_concat)
        data_concat.to_csv(f'{main_root}/data_time/{root[i]}.csv', index_label=False, index=False)


def image_visualization():
    data = root
    data.sort()
    for i in data:
        # 경로 자기 디렉터리 순서로 바꿔주세요
        path_folder = f'{main_root}{file}{i}/1st/'
        path_folder2 = f'{main_root}img_direct/{i}/'
        print(path_folder2)
        root_data = os.listdir(path_folder)
        root_data.sort()
        for j in root_data:
            a = pd.read_csv(path_folder+j)
            col = a.columns
            ax = plt.gca()

            ax.set_facecolor('white')
            plt.plot(a[col[0]], a[col[1]], c='black')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.axis('off')
            plt.savefig(f'{path_folder2}/{i}__{j}.png', facecolor='white')
            plt.show()


image_visualization()