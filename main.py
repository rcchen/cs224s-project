import os

import tensorflow as tf
import numpy as np

flags = tf.app.flags

flags.DEFINE_string('model', '', 'The name of the model to run.')
flags.DEFINE_string('data_dir', 'data/', 'The directory containing data files.')
flags.DEFINE_string('mode', 'train', 'Running mode: either "train", "dev", or "test."')

# TODO: Add hyperparameter flags

# TODO: Add training flags

FLAGS = flags.FLAGS


def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # TODO: Train the model on the data.
        raise NotImplementedError


def test():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # TODO: Evaluate the model on the data.
        raise NotImplementedError


def main(unused_argv):
    with tf.Graph().as_default():
        # TODO: Build the specified model and run it on the specified mode.
        raise NotImplementedError


if __name__ == "__main__":
    tf.app.run()
