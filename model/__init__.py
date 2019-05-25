
import tensorflow as tf
import os

import model.network as network


class Model(object):

    def __init__(self, config, batch_llen=None):

        self.lr = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='low_resolution')
        self.gt = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='ground_truth')

        # with tf.device(config.device):
        self.inference, self.residual, self.weights, self.biases = network.inference(self.lr, config.scale)

        self.saver = tf.train.Saver(self.weights + self.biases)

        if config.is_train:
            self.global_step = tf.Variable(0, trainable=False)

            with tf.name_scope('total_loss'):
                self.loss, self.regularizer = self._loss_function(config.reg_parameter, config.pretrain)

            with tf.name_scope('vdsr_learning_rate'):
                if config.learning_rate_decay:
                    self.learning_rate = tf.train.exponential_decay(
                        config.learning_rate, self.global_step,
                        config.decay_step * batch_llen, config.decay_rate,
                        staircase=True
                    )
                else:
                    self.learning_rate = config.learning_rate

            with tf.name_scope('vdsr_optimizer'):
                self.optimize = self._optimization_function()

            self.psnr = tf.reduce_mean(tf.image.psnr(self.gt, self.inference, 1.))
            self.mse = tf.reduce_mean(tf.losses.mean_squared_error(self.gt, self.inference))

    def _loss_function(self, reg_parameter, pretrain):
        if pretrain:
            loss = tf.reduce_sum(tf.abs(self.gt - self.inference), name='idn_loss')
        else:
            loss = tf.nn.l2_loss(self.gt - self.inference, name='idn_loss')

        regularizer = 0
        for w in self.weights:
            regularizer += tf.multiply(tf.nn.l2_loss(w), reg_parameter, name='regularization')

        loss = tf.reduce_mean(loss + regularizer)

        return loss, regularizer

    def _optimization_function(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        return optimizer

    def save(self, session, checkpoint_path, model_name, step):

        path = os.path.join(checkpoint_path, model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        self.saver.save(session, os.path.join(path, model_name), global_step=step)

    def load(self, session, checkpoint_path, model_name, pretrain=False, pretrained_model_name=None):

        if pretrain:
            name = pretrained_model_name
        else:
            name = model_name

        path = os.path.join(checkpoint_path, name)
        checkpoint = tf.train.get_checkpoint_state(path)

        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(session, os.path.join(path, checkpoint_name))
            return True
        else:
            return False
