import tensorflow as tf
import os

import model.network as network


class Model(object):

    def __init__(self, config):
        self.model_config = config

        image_size = self.model_config.image_size
        channels = self.model_config.channels

        # placeholders for image and labels
        self.image = tf.placeholder(tf.float32, shape=(None, image_size, image_size, channels))
        self.label = tf.placeholder(tf.float32, shape=(None, image_size, image_size, channels))

        # inference
        self.inference, self.feature, self.ilr, self.weights, self.biases = network.inference(self.image, config.scale)

        # loss function
        if not config.pretrain:
            self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.label, self.inference))
        else:
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.label - self.inference))

        # regularization term
        self.regularizer = 0
        for w in self.weights:
            self.regularizer += tf.nn.l2_loss(w)

        self.loss += self.regularizer * 1e-4

        self.psnr = tf.reduce_mean(tf.image.psnr(self.label, self.inference, max_val=1.))
        self.mse = tf.losses.mean_squared_error(self.label, self.inference)
        self.global_step = tf.Variable(0, trainable=False)

        if config.decay_learning_rate:
            decay_step = config.decay_step
            decay_rate = config.decay_rate

            self.learning_rate = tf.train.exponential_decay(
                config.learning_rate, self.global_step,
                decay_step, decay_rate,
                staircase=True
            )
        else:
            self.learning_rate = config.learning_rate

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(self.weights + self.biases)

    def save(self, session, step):

        path = os.path.join(self.model_config.checkpoint_path, self.model_config.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        self.saver.save(session, os.path.join(path, self.model_config.model_name), global_step=step)

    def load(self, session):

        if self.model_config.pretrain:
            name = self.model_config.pretrained_model_name
        else:
            name = self.model_config.model_name

        path = os.path.join(self.model_config.checkpoint_path, name)
        checkpoint = tf.train.get_checkpoint_state(path)

        if checkpoint and checkpoint.model_checkpoint_path:

            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(session, os.path.join(path, checkpoint_name))
            return True
        else:
            return False
