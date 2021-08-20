import numpy as np
import pandas as pd

data = 'test1111.csv'

def sss():
    x_all = []
    y_all = []
    for i in range(0, len(data)):
        csv_data = pd.read_csv(data)
        zeros_mat = np.zeros(csv_data.shape)
        j = 0

        while j <= 40:
            a = zeros_mat[:, j:j+1] = j+1
            j += 1

        x_all.append(np.array(csv_data))
        y_all.append(np.array(zeros_mat))

    return x_all, y_all


def compile_all_except(all_, i_):
    compiled = np.empty([0, *all_[0].shape[1:]], np.float32)
    for i in range(0, len(all_)):
        if i != i_:
            compiled = np.concatenate((compiled, all_[i]), axis=0)
    return compiled


x_all, y_all = sss()
acc = np.empty([len(x_all)]).astype(np.float32)
for i in range(0, len(x_all)):
    x_test = x_all[i]
    y_test = y_all[i]
    X_train = compile_all_except(x_all, i)
    y_train = compile_all_except(y_all, i)

    print(X_train.shape)
    print(y_train.shape)

