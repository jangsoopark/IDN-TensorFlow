
import tensorflow as tf

import skimage.measure as measure
import scipy.io

import numpy as np

import os

import model
import utils
import data


def run(config):

    test_data_path = './data/test_data/mat/'
    result_root = './result/'

    benchmark_list = ['Set5', 'Set14', 'B100', 'Urban100']
    scale = [2, 3, 4]

    if not os.path.exists(result_root):
        os.makedirs(result_root)
        for benchmark in benchmark_list:
            os.makedirs(os.path.join(result_root, benchmark))
            for s in scale:
                os.makedirs(os.path.join(result_root, benchmark, str(s)))

    s = config.scale
    with tf.Session() as sess:
        vdsr = model.Model(config)
        vdsr.load(sess, config.checkpoint_path, config.model_name)

        for benchmark in benchmark_list:
            print(benchmark)
            test_benchmark_path = os.path.join(test_data_path, benchmark)

            lr, gt = data.load_lr_gt_mat(test_benchmark_path, s)

            quality_result = open(
                os.path.join(result_root, benchmark, 'quality_%d.csv' % s), 'w'
            )

            quality_result.write('file name, psnr, ssim\n')
            psnr_list = []
            ssim_list = []
            for i, _ in enumerate(gt):

                lr_image = lr[i]['data']
                gt_image = gt[i]['data']

                sr = sess.run(vdsr.inference, feed_dict={
                    vdsr.lr: lr_image.reshape((1,) + lr_image.shape + (1,))
                })

                sr = sr.reshape(sr.shape[1: 3])

                sr_ = utils.shave(sr, s)
                sr_ = sr_.astype(np.float64)
                gt_image_ = utils.shave(gt_image, s)

                _psnr = measure.compare_psnr(gt_image_, sr_)
                _ssim = measure.compare_ssim(gt_image_, sr_)

                quality_result.write('%s, %f, %f\n' % (gt[i]['name'], _psnr, _ssim))
                psnr_list.append(_psnr)
                ssim_list.append(_ssim)

                scipy.io.savemat(
                    os.path.join(result_root, benchmark, str(s), gt[i]['name'] + '.mat'), {'sr': sr}
                )


            quality_result.close()