

import numpy as np
import scipy.io
import h5py

import glob
import os


def load_h5(file_name):

    with h5py.File(file_name, 'r') as data:
        lr = np.asarray(data.get('lr'), dtype=np.float64)
        gt = np.asarray(data.get('gt'), dtype=np.float64)

    lr_shape = lr.shape
    gt_shape = gt.shape

    lr = lr.reshape((lr_shape[0], lr_shape[2], lr_shape[3], lr_shape[1]))
    gt = gt.reshape((gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[1]))

    return lr, gt


def load_lr_gt_mat(file_path, scale):

    gt_path_list = glob.glob(
        os.path.join(file_path, '*_%d_%s.mat' % (scale, 'gt'))
    )

    lr_path_list = glob.glob(
        os.path.join(file_path, '*_%d_%s.mat' % (scale, 'lr'))
    )

    gt_list = []
    lr_list = []

    for i in range(len(gt_path_list)):
        name = gt_path_list[i].split('\\')[-1].split('.')[0].split('_')[0]

        gt = scipy.io.loadmat(gt_path_list[i])['gt'].astype(np.float64)
        lr = scipy.io.loadmat(lr_path_list[i])['lr'].astype(np.float64)

        gt_list.append({'name': name, 'data': gt})
        lr_list.append({'name': name, 'data': lr})

    return lr_list, gt_list
