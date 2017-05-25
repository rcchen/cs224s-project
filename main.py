import os

import tensorflow as tf
import numpy as np

from var.utils.vocab import Vocab

flags = tf.app.flags

flags.DEFINE_string('model', '', 'The name of the model to run.')
flags.DEFINE_string('data_dir', 'var/data/', 'The directory containing data files.')
flags.DEFINE_string('mode', 'train', 'Running mode: either "train", "dev", or "test."')
flags.DEFINE_string('vocab_file', 'vocab.txt', 'The file containing a list of vocab.')

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

    # Load the vocabulary file.
    vocab = Vocab(FLAGS.vocab_file, os.path.join(FLAGS.data_dir, 'essays/dev'))

    with tf.Graph().as_default():
        # TODO: Build the specified model and run it on the specified mode.
        raise NotImplementedError


if __name__ == "__main__":
    tf.app.run()
