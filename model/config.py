import tensorflow as tf

config = tf.app.flags

config.DEFINE_string('model_name', 'idr', 'name of checkpoint directory')
config.DEFINE_string('checkpoint_path', 'checkpoint', 'name of checkpoint directory')

config.DEFINE_string('device', '/gpu:1', 'device name')
config.DEFINE_boolean('pretrain', False, 'use pre-trained model')
config.DEFINE_string('pretrained_model_name', 'idr', 'use pre-trained model')

config.DEFINE_integer('epochs', 100000, 'number of epochs 1e+5 for pre training 6e+5 for training')
config.DEFINE_integer('scale', 2, 'size of input image')
config.DEFINE_integer('image_size', None, 'size of input image')
config.DEFINE_integer('channels', 1, 'number of channels')
config.DEFINE_integer('batch_size', 64, 'number of images in mini batch')

config.DEFINE_boolean('is_train', True, '[True] if train')

config.DEFINE_float('learning_rate', 1e-4, 'learning rate')

config.DEFINE_boolean('decay_learning_rate', False, '[True] if train')
config.DEFINE_float('decay_rate', 1e-1, 'learning rate decay step')
config.DEFINE_integer('decay_step', 250000, 'number of images in mini batch')