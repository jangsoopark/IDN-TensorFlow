import tensorflow as tf

he_init = tf.initializers.he_normal


def convolution(image, shape, scope, name, index):
    with tf.name_scope('%s_%s_%d' % (scope, name, index)) as scope:
        weight = tf.get_variable('weight_%s_%d' % (name, index), shape, initializer=he_init(),
                                 dtype=tf.float32)
        bias = tf.get_variable('bias_%s_%d' % (name, index), shape[- 1], initializer=tf.zeros_initializer(),
                               dtype=tf.float32)
        conv = tf.nn.conv2d(
            image, weight, strides=[1, 1, 1, 1], padding='SAME', name='conv_%s_%d' % (name, index)
        ) + bias
    return conv, weight, bias


def grouped_convolution(image, group, shape, scope, name, index, block_index):
    split_shape = [shape[2] // group for _ in range(group)]
    grouped_feature = tf.split(image, split_shape, axis=3)

    layers = []
    weights = []
    biases = []
    ind = 0
    for feature in grouped_feature:
        layer, weight, bias = convolution(
            feature, [3, 3, shape[2] // group, shape[3] // group],
            'information_distillation_block',
            'dblock_eunit_%d_group_%d' % (block_index, ind), index
        )
        layers.append(layer)
        weights.append(weight)
        biases.append(bias)
        ind += 1

    conv = tf.concat(layers, axis=3)
    return conv, weights, biases


# number of input channels : 1
# number of output channels : 64
def feature_extraction_block(image):
    weights = []
    biases = []
    feature_0, weight, bias = convolution(image, [3, 3, 1, 64], 'feature_extraction', 'fblock', 0)
    feature_0 = tf.nn.leaky_relu(feature_0, 0.05)
    weights.append(weight)
    biases.append(bias)

    feature_1, weight, bias = convolution(feature_0, [3, 3, 64, 64], 'feature_extraction', 'fblock', 1)
    feature_1 = tf.nn.leaky_relu(feature_1, 0.05)
    weights.append(weight)
    biases.append(bias)

    return feature_1, weights, biases


# number of input channels : 64
# number of output channels : 80
def enhancement_unit(features, block_index):
    weights = []
    biases = []

    layer_0, weight, bias = convolution(
        features, [3, 3, 64, 48], 'information_distillation_block', 'dblock_eunit_%d' % block_index, 0)
    layer_0 = tf.nn.leaky_relu(layer_0, 0.05)
    weights.append(weight)
    biases.append(bias)

    layer_1, weight, bias = grouped_convolution(
        layer_0, 4, [3, 3, 48, 32], 'information_distillation_block', 'dblock_eunit_%d', 1, block_index)
    layer_1 = tf.nn.leaky_relu(layer_1, 0.05)
    weights += weight
    biases += bias

    layer_2, weight, bias = convolution(
        layer_1, [3, 3, 32, 64], 'information_distillation_block', 'dblock_eunit_%d' % block_index, 2)
    layer_2 = tf.nn.leaky_relu(layer_2, 0.05)
    weights.append(weight)
    biases.append(bias)

    segment_1, segment_2 = tf.split(layer_2, [16, 48], axis=3)
    concatenated = tf.concat([features, segment_1], axis=3)

    layer_3, weight, bias = convolution(
        segment_2, [3, 3, 48, 64], 'information_distillation_block', 'dblock_eunit_%d' % block_index, 3)
    layer_3 = tf.nn.leaky_relu(layer_3, 0.05)
    weights.append(weight)
    biases.append(bias)

    layer_4, weight, bias = grouped_convolution(
        layer_3, 4, [3, 3, 64, 48], 'information_distillation_block', 'dblock_eunit_%d', 4, block_index)
    layer_4 = tf.nn.leaky_relu(layer_4, 0.05)
    weights += weight
    biases += bias

    layer_5, weight, bias = convolution(
        layer_4, [3, 3, 48, 80], 'information_distillation_block', 'dblock_eunit_%d' % block_index, 5)
    layer_5 = tf.nn.leaky_relu(layer_5, 0.05)
    weights.append(weight)
    biases.append(bias)

    return layer_5 + concatenated, weights, biases


# number of input channels : 80
# number of output channels : 64
def compression_unit(features, block_index):
    weights = []
    biases = []

    layer, weight, bias = convolution(
        features, [1, 1, 80, 64], 'information_distillation_block', 'dblock_cunit_%d' % block_index, 0)
    layer = tf.nn.leaky_relu(layer, 0.05)
    weights.append(weight)
    biases.append(bias)

    return layer, weights, biases


# number of input channels : 64
# number of output channels : 64
# {input features, 64} -> enhancement unit -> compression unit -> {output features, 64}
def information_distillation_block(features, block_index):
    weights = []
    biases = []

    enhanced_features, weight, bias = enhancement_unit(features, block_index)
    weights += weight
    biases += bias

    compressed_features, weight, bias = compression_unit(enhanced_features, block_index)
    weights += weight
    biases += bias
    return compressed_features, weights, biases


# number of input channels : 64
# number of output channels : 1
def reconstruction_block(features, scale):
    weights = []
    biases = []

    shape = tf.shape(features)
    output_shape = [shape[0], scale * shape[1], scale * shape[2], 1]
    stride = [1, scale, scale, 1]

    with tf.name_scope('reconstruction_block') as scope:

        weight = tf.get_variable('weight_reconstruction', [17, 17, 1, 64],
                                 initializer=he_init(),
                                 dtype=tf.float32)
        bias = tf.get_variable('bias_reconstruction', output_shape[-1],
                               initializer=tf.zeros_initializer(),
                               dtype=tf.float32)

        reconstructed_feature = tf.nn.conv2d_transpose(features, weight, output_shape, stride, padding='SAME') + bias
    weights.append(weight)
    biases.append(bias)

    return reconstructed_feature, weights, biases


def inference(image, scale):
    weights = []
    biases = []

    feature, weight, bias = feature_extraction_block(image)
    weights += weight
    biases += bias

    for i in range(4):
        feature, weight, bias = information_distillation_block(feature, i)
        weights += weight
        biases += bias

    feature, weight, bias = reconstruction_block(feature, scale)
    weights += weight
    biases += bias

    shape = tf.shape(image)[1:3]
    interpolated_low_resolution = tf.image.resize_images(
        image,
        (scale * shape[0], scale * shape[1]),
        method=tf.image.ResizeMethod.BICUBIC
    )
    output = feature + interpolated_low_resolution
    return output, feature, interpolated_low_resolution, weights, bias
