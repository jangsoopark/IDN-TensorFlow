
import tensorflow as tf

import model.configurations as configurations
import train
import test


def main(_):
    config = configurations.config.FLAGS

    if config.is_train:
        train.run(config)
    else:
        test.run(config)


if __name__ == '__main__':
    tf.app.run()
