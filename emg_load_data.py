import os
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

def load_data(file='test_image_data.npz'):
    root_location = os.getcwd()
    path = get_file(
        file,
        origin=root_location + file

    )
    with np.load(path, allow_pickle=True) as f:
        x_train, x_test = f['x_train'], f['x_test']
        y_train, y_test = f['y_train'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
