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
    # TODO: Returns the correct corresponding model. Let's start with just SVM.

    # BASELINE DETAILS
    # Baseline uses tf.contrib.learn.SVM, a well-defined classifier. It takes in
    # FeatureColumns, and we just call `fit` and `evaluate` on the classifier
    # instance.

    # TRAINING AND EVALUATING SVM
    # `fit` represents an instance of training, and `evaluate` gives us both the
    # loss and accuracy metrics. It uses SDCAOptimizer by default.

    # DATA VECTORIZATION
    # We will also need to implement something like a CountVectorizer. Right now
    # our dataset wraps around a pandas DataFrame containing vocab indices of
    # tokens from our dataset. While this will help us train our models on
    # sequence-aware neural nets, SVM will need to use token counts instead.

    # More information about SVM
    # Documentation: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/SVM
    # Example usage: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/learn/python/learn/estimators/svm_test.py

    return NotImplementedError


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
