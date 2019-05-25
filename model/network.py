
import tensorflow as tf

import utils


def feature_extraction_block(input_):

    weights = []
    biases = []

    layer, weight, bias = utils.convolution(
        input_,
        shape=[3, 3, 1, 64],
        strides=[1, 1, 1, 1],
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='fblock', index=0
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    layer, weight, bias = utils.convolution(
        layer,
        shape=[3, 3, 64, 64],
        strides=[1, 1, 1, 1],
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='fblock', index=1
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    return layer, weights, biases


def enhancement_unit(input_, block_index):
    weights = []
    biases = []

    layer, weight, bias = utils.grouped_convolution(
        input_,
        shape=[3, 3, 64, 48],
        strides=[1, 1, 1, 1],
        padding='SAME',
        group=4,
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='enhancement_unit_%d' % block_index, index=0
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights += weight
    biases += bias

    layer, weight, bias = utils.convolution(
        layer,
        shape=[3, 3, 48, 32],
        strides=[1, 1, 1, 1],
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='enhancement_unit_%d' % block_index, index=1
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    layer, weight, bias = utils.convolution(
        layer,
        shape=[3, 3, 32, 64],
        strides=[1, 1, 1, 1],
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='enhancement_unit_%d' % block_index, index=2
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    segment_1, segment_2 = tf.split(layer, [16, 48], axis=3)
    concatenated = tf.concat([input_, segment_1], axis=3)

    layer, weight, bias = utils.grouped_convolution(
        segment_2,
        shape=[3, 3, 48, 64],
        strides=[1, 1, 1, 1],
        padding='SAME',
        group=4,
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='enhancement_unit_%d' % block_index, index=3
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights += weight
    biases += bias

    layer, weight, bias = utils.convolution(
        layer,
        shape=[3, 3, 64, 48],
        strides=[1, 1, 1, 1],
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='enhancement_unit_%d' % block_index, index=4
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    layer, weight, bias = utils.convolution(
        layer,
        shape=[3, 3, 48, 80],
        strides=[1, 1, 1, 1],
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='enhancement_unit_%d' % block_index, index=5
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    output = layer + concatenated

    return output, weights, biases


def compression_unit(input_, block_index):
    weights = []
    biases = []

    layer, weight, bias = utils.convolution(
        input_,
        shape=[1, 1, 80, 64],
        strides=[1, 1, 1, 1],
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='compression_unit_%d' % block_index, index=0
    )
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    return layer, weights, biases


def information_distillation_block(input_, block_index):
    weights = []
    biases = []

    with tf.name_scope('dblock_%d' % block_index):
        layer, weight, bias = enhancement_unit(input_, block_index)
        weights += weight
        biases += bias

        layer, weight, bias = compression_unit(layer, block_index)
        weights += weight
        biases += bias

    return layer, weights, biases


def reconstruction_block(input_, scale):
    weights = []
    biases = []

    shape = tf.shape(input_)
    output_shape = [shape[0], scale * shape[1], scale * shape[2], 1]
    stride = [1, scale, scale, 1]

    layer, weight, bias = utils.deconvolution(
        input_,
        shape=[17, 17, 1, 64],
        output_shape=output_shape,
        stride=stride,
        padding='SAME',
        init_w=tf.initializers.he_normal(),
        init_b=tf.initializers.zeros(),
        name='rblock', index=0
    )

    weights.append(weight)
    biases.append(bias)
    return layer, weights, biases


def inference(image, scale):

    weights = []
    biases = []

    layer, weight, bias = feature_extraction_block(image)
    weights += weight
    biases += bias

    for i in range(4):
        layer, weight, bias = information_distillation_block(layer, i)
        weights += weight
        biases += bias

    layer, weight, bias = reconstruction_block(layer, scale)
    weights += weight
    biases += bias

    with tf.name_scope('bicubic_interpolation'):
        shape = tf.shape(image)[1:3]
        ilr = tf.image.resize_images(
            image,
            (scale * shape[0], scale * shape[1]),
            method=tf.image.ResizeMethod.BICUBIC
        )

    with tf.name_scope('output'):
        output = layer + ilr
        output = tf.clip_by_value(output, 0., 1.)

    return output, layer, weights, biases

