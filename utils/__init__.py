from PIL import Image

import tensorflow as tf

import numpy as np
import math


def convolution(input_, shape, strides, padding, init_w, init_b, name, index):

    weight = tf.get_variable(
        '%s_weight_%d' % (name, index), shape=shape, initializer=init_w, dtype=tf.float32
    )
    bias = tf.get_variable(
        '%s_bias_%d' % (name, index), shape=shape[-1], initializer=init_b, dtype=tf.float32
    )

    conv = tf.nn.conv2d(
        input_, weight, strides, padding, name='%s_conv2d_%d' % (name, index)
    ) + bias

    return conv, weight, bias


def deconvolution(input_, shape, output_shape, stride, padding, init_w, init_b, name, index):

    weight = tf.get_variable('weight_reconstruction', shape,
                             initializer=init_w,
                             dtype=tf.float32)
    bias = tf.get_variable('bias_reconstruction', output_shape[-1],
                           initializer=init_b,
                           dtype=tf.float32)

    conv = tf.nn.conv2d_transpose(
        input_, weight, output_shape, stride, padding, name='%s_conv2d_%d' % (name, index)
    ) + bias
    return conv, weight, bias


def grouped_convolution(input_, shape, strides, padding, group, init_w, init_b, name, index):

    split_shape = [shape[2] // group for _ in range(group)]
    grouped_input = tf.split(input_, split_shape, axis=3)

    layers = []
    weights = []
    biases = []

    i = 0
    for __input in grouped_input:
        layer, weight, bias = convolution(
            __input,
            shape=[3, 3, shape[2] / group, shape[3] // group], strides=strides,
            padding=padding,init_w=init_w,init_b=init_b,
            name='%s_%d' % (name, i), index=index
        )
        i += 1
        layers.append(layer)
        weights.append(weight)
        biases.append(bias)
    conv = tf.concat(layers, axis=3)
    return conv, weights, biases


def shave(image, scale):
    return image[scale: -scale, scale: -scale]


def psnr(gt, sr):

    diff = gt - sr
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2))

    return 20 * math.log10(1. / rmse)


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def save_image(image, path):

    image = image * 255
    image = image.clip(0, 255).astype(np.uint8)
    Image.fromarray(image, mode='L').convert('RGB').save(path)