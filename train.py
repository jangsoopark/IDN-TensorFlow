
import tensorflow as tf

import numpy as np
import os

import utils
import model
import data


def summary(sess, idn, model_name):

    tf.summary.image('input_image', idn.lr, max_outputs=8)
    tf.summary.image('label_image', idn.gt, max_outputs=8)
    tf.summary.image('residual_image', idn.residual, max_outputs=8)
    tf.summary.image('output_image', idn.inference, max_outputs=8)

    tf.summary.scalar('learning_rate', idn.learning_rate)
    tf.summary.scalar('loss/loss', idn.loss)
    tf.summary.scalar('loss/regularizer', idn.regularizer)
    tf.summary.scalar('loss/mse', idn.mse)
    tf.summary.scalar('loss/psnr', idn.psnr)

    for i in range(20):
        tf.summary.histogram('01-weights/layer_%d' % i, idn.weights[i])
        tf.summary.histogram('02-bias/layer_%d' % i, idn.biases[i])

    merged = tf.summary.merge_all()
    writter = tf.summary.FileWriter('./summary/' + model_name, sess.graph)

    return merged, writter


def validation(sess, vdsr, epoch, scale):

    if not os.path.exists('./validation'):
        os.makedirs('./validation')

    validation_result_path = {
        2: './validation/2.csv',
        3: './validation/3.csv',
        4: './validation/4.csv'
    }

    s = scale
    if not os.path.exists('./validation/%d' % s):
        os.makedirs('./validation/%d' % s)

    lr, gt = data.load_lr_gt_mat('./data/test_data/mat/Set5', s)
    v_len = len(gt)

    psnr = []

    for i in range(v_len):
        lr_image = lr[i]['data']
        gt_image = gt[i]['data']

        residual, sr = sess.run([vdsr.residual, vdsr.inference], feed_dict={
            vdsr.lr: lr_image.reshape((1,) + lr_image.shape + (1,))
        })

        sr = sr.reshape(sr.shape[1: 3])
        residual = residual.reshape(residual.shape[1: 3])

        utils.save_image(sr, './validation/%d/%s_sr_scale_%d_epoch_%d.png' % (s, lr[i]['name'], s, epoch))

        residual = utils.normalize(residual)
        utils.save_image(
            residual, './validation/%d/%s_residual_scale_%d_epoch_%d.png' % (s, lr[i]['name'], s, epoch))

        sr_ = utils.shave(sr, s)
        gt_image_ = utils.shave(gt_image, s)
        psnr.append(utils.psnr(gt_image_, sr_))
    with open(validation_result_path[s], 'a') as f:
        f.write('%d, %s, %f\n' % (epoch, ', '.join(str(e) for e in psnr), float(np.mean(psnr))))


def run(config):

    lr, gt = data.load_h5(config.data_path)
    batch_len = lr.shape[0] // config.batch_size

    with tf.Session() as sess:
        idn = model.Model(config, batch_len)

        step = 0
        merged, writer = summary(sess, idn, config.model_name)

        sess.run(tf.global_variables_initializer())
        if config.pretrain:
            idn.load(sess, config.checkpoint_path, config.model_name, config.pretrain, config.pretrained_model_name)

        for i in range(config.epochs):
            print('Epoch = %d' % i)
            batch_index = zip(
                range(0, lr.shape[0], config.batch_size),
                range(config.batch_size, lr.shape[0] + 1, config.batch_size),
            )
            for s, e in batch_index:
                _, loss, psnr = sess.run([idn.optimize, idn.loss, idn.psnr], feed_dict={
                    idn.lr: lr[s: e], idn.gt: gt[s: e]
                })
                print('Epoch = %d, batch = %d / %d, loss = %.4f, psnr = %.4f' % (
                    i, s, lr.shape[0], loss, psnr
                ))

                if step % 100 == 0:
                    summary_ = sess.run(merged, feed_dict={idn.lr: lr[s: e], idn.gt: gt[s: e]})
                    writer.add_summary(summary_, step)
                step += 1
            validation(sess, idn, i, config.scale)
            idn.save(sess, config.checkpoint_path, config.model_name, step)