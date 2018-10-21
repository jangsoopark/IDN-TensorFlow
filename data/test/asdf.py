
import scipy.io

import glob
import os

scale = 2
ext = '.mat'

file_list = glob.glob('./benchmark/Set5/*_%d.mat' % scale)

for file_name in file_list:
    hr = file_name.split('_')[0] + ext
    lr = file_name
    print(hr, lr)
