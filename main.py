import tensorflow as tf

import model.config as config
import model.train as train
import model.test as test


def main(_):
    model_config = config.config.FLAGS

    if model_config.is_train:
        train.trainer(model_config)
    else:
        test.test(model_config)


if __name__ == '__main__':
    tf.app.run()