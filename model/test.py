
from PIL import Image

import tensorflow as tf

import numpy as np
import scipy.io

import time
import glob
import os

import model.model as model
import matlab.engine


def test(config):
    data_path = './data/test/benchmark/'
    file_name_ = '/*_%d'
    result_path = './result/'

    benchmark_list = ['B100', 'Set5', 'Set14', 'Urban100']
    ext = ".mat"

    benchmark_len = len(benchmark_list)

    sr_model = model.Model(config)
    engine = matlab.engine.start_matlab()

    with tf.Session() as sess:
        sr_model.load(sess)

        for i in range(benchmark_len):
            print(benchmark_list[i])

            result_benchmark_path = os.path.join(result_path, benchmark_list[i])
            if not os.path.exists(result_benchmark_path):
                os.makedirs(result_benchmark_path)

            for scale in [config.scale]:
                print(scale)
                file_list = glob.glob(data_path + benchmark_list[i] + file_name_ % scale + ext)
                result_image_path = os.path.join(result_path, benchmark_list[i], str(scale))
                if not os.path.exists(result_image_path):
                    os.makedirs(result_image_path)

                time_f = open(os.path.join(result_benchmark_path, 'time_%d.csv' % scale), 'w')
                psnr_f = open(os.path.join(result_benchmark_path, 'psnr_%d.csv' % scale), 'w')
                s = float(scale)

                for file_name in file_list:
                    hr_file_name = file_name.split('_')[0] + ext
                    hr = scipy.io.loadmat(hr_file_name)['img_raw'].astype(np.float32)
                    lr = scipy.io.loadmat(file_name)['img_%d' % scale].astype(np.float32)

                    start_time = time.time()
                    sr = sess.run(sr_model.inference, feed_dict={
                        sr_model.image: lr.reshape((-1,) + lr.shape + (1, ))
                    })
                    end_time = time.time()

                    sr = sr.reshape(hr.shape)

                    psnr, ssim = engine.compute_diff(matlab.double(hr.tolist()), matlab.double(sr.tolist()), s, nargout=2)

                    print('PSNR : %f, SSIM : %f' % (psnr, ssim))

                    time_per_image = end_time - start_time

                    time_f.write('%s, %.8f\n' % (file_name.split('\\')[-1], time_per_image))
                    psnr_f.write('%s, %.8f, %.8f\n' % (
                        file_name.split('\\')[-1], psnr, ssim))

                    sr = sr * 255
                    sr = np.round(sr.clip(0, 255)).astype(np.uint8)
                    sr_out = np.zeros(sr.shape + (3, ), dtype=np.uint8)
                    sr_out[:, :, 0] = sr

                    sr_out[:, :, 1] = sr
                    sr_out[:, :, 2] = sr
                    sr_ = Image.fromarray(sr_out).convert('RGB')

                    sr_.save(result_image_path + '/' + file_name.split('\\')[-1].split('.')[0] + '.png')
                time_f.close()
                psnr_f.close()

    engine.quit()
