
import numpy as np
import glob

import h5py


def load_data_set(path, extension):
    file_list = glob.glob(path + '*.' + extension)

    data_list = []
    for file_name in file_list:
        data = np.load(file_name)
        data_list += data.tolist()

    return np.asarray(data_list, dtype=np.float32)


def load_np_data_set(file_name):

    data = np.load(file_name)

    return data


def load_np_data_as_list(file_name):
    data_list = []

    data = np.load(file_name)
    data_list = data.tolist()

    return data_list


def load_h5(file_name):

    with h5py.File(file_name, 'r') as data:
        images = np.asarray(data.get('data'))
        labels = np.asarray(data.get('label'))

    shape_input = images.shape
    shape_label = labels.shape

    images = images.reshape((shape_input[0], shape_input[2], shape_input[3], shape_input[1]))
    labels = labels.reshape((shape_label[0], shape_label[2], shape_label[3], shape_label[1]))

    return images, labels

