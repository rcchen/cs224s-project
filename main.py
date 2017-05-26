import os

import tensorflow as tf
import numpy as np

import var.models

from var.utils.dataset import Dataset
from var.utils.progbar import Progbar
from var.utils.vocab import Vocab

flags = tf.app.flags

flags.DEFINE_string('model', 'baseline', 'The name of the model to run.')
flags.DEFINE_string('mode', 'train', 'Running mode: either "dev", "train", or "test"')
flags.DEFINE_string('data_dir', 'var/data', 'The directory containing data files.')
flags.DEFINE_string('input_type', 'essays', 'Input data feature type: either "essays", \
                    "speech_transcriptions", "ivectors", or \
                    "speech_transcriptions+ivectors" ')
flags.DEFINE_string('preprocessor', 'tokenized', 'Name of directory with processed essay files.')
flags.DEFINE_string('max_seq_len', 1000, 'Max number of words in an example.')
flags.DEFINE_string('debug', False, 'Run on debug mode, using a smaller data set.')

FLAGS = flags.FLAGS

vocab_file = os.path.join(FLAGS.data_dir, 'vocab.txt')
regular_data_file = os.path.join(FLAGS.data_dir, 'data.pkl')
debug_data_file = os.path.join(FLAGS.data_dir, 'debug_data.pkl')


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
    vocab = Vocab(vocab_file, os.path.join(FLAGS.data_dir, FLAGS.input_type))

    # Load the data file.
    dataset = Dataset(FLAGS.data_dir, FLAGS.input_type, FLAGS.preprocessor,
                      FLAGS.mode, FLAGS.max_seq_len, vocab, regular_data_file,
                      debug_data_file, FLAGS.debug)

    # Load the model.
    model = get_model()
    model.build()

    with tf.Graph().as_default():
        # TODO: Build the specified model and run it on the specified mode.
        raise NotImplementedError


if __name__ == "__main__":
    tf.app.run()
