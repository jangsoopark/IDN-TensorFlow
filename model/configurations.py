
import tensorflow as tf


config = tf.app.flags


config.DEFINE_bool('is_train', False, 'set True for train')

# model name
config.DEFINE_string('model_name', 'idn', 'model name')
config.DEFINE_string('checkpoint_path', 'checkpoint', 'checkpoint directory')

# if you want to run this implementation on CPU, set the second parameter as '/cpu:0'
# config.DEFINE_string('device', '/gpu:0', 'device for operation')

# training configuration

config.DEFINE_string('data_path', './data/train_data/vdsr_train.h5', 'training dataset path')

config.DEFINE_integer('epochs', 80, 'maximum epochs for training')
config.DEFINE_integer('batch_size', 64, 'number of datas in mini batch')
config.DEFINE_float('learning_rate', 1e-4, 'learning rate for optimization')

# regularization parameter
config.DEFINE_float('reg_parameter', 1e-4, 'regularization parameter')

# - learning rate decay parameters
config.DEFINE_bool('learning_rate_decay', False, 'learning rate for optimization')
config.DEFINE_integer('decay_step', 250000, 'learning rate for optimization')
config.DEFINE_float('decay_rate', 1e-1, 'learning rate for optimization')

# pre train
config.DEFINE_bool('pretrain', False, 'use pre-trained model')
config.DEFINE_string('pretrained_model_name', 'idn_pre', 'pretrained model name')

# model scale
config.DEFINE_integer('scale', 2, 'low resolution scale')

