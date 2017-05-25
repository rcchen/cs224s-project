import os

import tensorflow as tf
import numpy as np

import var.models

from var.utils.dataset import Dataset
from var.utils.progbar import Progbar
from var.utils.vocab import Vocab

flags = tf.app.flags

flags.DEFINE_string('model', 'baseline', 'The name of the model to run.')
flags.DEFINE_string('data_dir', 'var/data', 'The directory containing data files.')
flags.DEFINE_string('features', 'essays', 'Input data feature type: either "essays", \
                    "speech_transcriptions", "ivectors", or \
                    "speech_transcriptions+ivectors" ')
flags.DEFINE_string('mode', 'train', 'Running mode: either "dev", "train", or "test"')
flags.DEFINE_string('vocab_file', 'vocab.txt', 'The file containing a list of vocab.')

# TODO: Add hyperparameter flags


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


def get_model():
    # Returns the correct corresponding model.
    # if FLAGS.model == 'baseline':
    #     return
    pass


def main(unused_argv):

    # Load the vocabulary file.
    vocab = Vocab(FLAGS.vocab_file, os.path.join(FLAGS.data_dir, FLAGS.features))

    # Load the data file.
    dataset = Dataset(os.path.join(FLAGS.data_dir, FLAGS.features, FLAGS.mode), vocab)

    # Load the model.
    model = get_model()
    model.build()

    with tf.Graph().as_default():
        # TODO: Build the specified model and run it on the specified mode.
        raise NotImplementedError


if __name__ == "__main__":
    tf.app.run()
