import numpy as np
from sklearn.metrics import confusion_matrix


def logw(file_, str_):
    print(str_)
    file_.write(str_ + '\n')


def maximize_output_probabilities_v2(array):
    array_out = np.copy(array)
    for i in range(array.shape[0]):
        b = np.zeros_like(array[i, :])
        b[array[i, :].argmax(0)] = 1
        array_out[i, :] = b
    return array_out


def print_confusion_matrix_v2(y_pred, y_true):
    yp2 = np.zeros([y_pred.shape[0], 1])
    yt2 = np.zeros([y_pred.shape[0], 1])
    y_pred = maximize_output_probabilities_v2(y_pred)
    for i in range(0, y_pred.shape[0]):
        yp2[i, :] = np.argmax(y_pred[i, :])
        yt2[i, :] = np.argmax(y_true[i, :])
    confusion = confusion_matrix(yt2, yp2)
    print('Confusion Matrix: \n', confusion)
    return confusion
