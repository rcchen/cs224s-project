import os

import tensorflow as tf
import numpy as np

from var.models import *
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
flags.DEFINE_string('batch_size', 100, 'Number of examples to run in a batch.')
flags.DEFINE_string('num_epochs', 50, 'Number of epochs to train for.')
flags.DEFINE_string('debug', False, 'Run on debug mode, using a smaller data set.')

# TODO: add model-saving capabilities
flags.DEFINE_boolean("save", True, "Whether to save the model.")

FLAGS = flags.FLAGS

vocab_file = os.path.join(FLAGS.data_dir, 'vocab.txt')
regular_data_file = os.path.join(FLAGS.data_dir, 'data.pkl')
debug_data_file = os.path.join(FLAGS.data_dir, 'debug_data.pkl')


def run_train_epoch(sess, model, dataset, epoch_num):
    print '='*79
    print 'Epoch: %s' % (epoch_num + 1)
    prog = Progbar(target = dataset.split_num_batches(FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_shuffled_iterator(FLAGS.batch_size)):
        loss, summary = model.train_on_batch(sess, *batch)
        prog.update(i + 1, [('train loss', loss)])
    print '='*79


def run_eval_epoch(sess, model, dataset):
    batch_sizes = []
    accuracies = []
    preds = []

    print '-'*79
    prog = Progbar(target=dataset.split_num_batches(FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_iterator(FLAGS.batch_size)):
        acc, loss, pred = model.evaluate_on_batch(sess, *batch)
        prog.update(i + 1, [('%s loss' % loss)])

        batch_sizes.append(batch[0].shape[0])
        accuracies.append(acc)
        preds.append(pred)

    accuracy = np.average(accuracies, weights=batch_sizes)
    print 'Accuracy: %s' % accuracy
    print '-'*79
    return accuracy, np.concatenate(preds)


def train(model, dataset):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        best_accuracy = 0
        for epoch in range(FLAGS.num_epochs):
            run_train_epoch(sess, model, dataset, epoch)
            # TODO: evaluate on a split (train or dev)
            # dev_accuracy, _ = run_eval_epoch(sess, model, dataset)
            if dev_accuracy > best_accuracy:
                # TODO: Save the model, as it's optimal.
                best_accuracy = dev_accuracy


def test(model, dataset):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # TODO: Evaluate the model on the data.
        raise NotImplementedError


def get_model(vocab, dataset):
    kwargs = {
        'batch_size': FLAGS.batch_size,
        'max_seq_len': FLAGS.max_seq_len,
        'num_classes': len(dataset.CLASS_LABELS)
    }
    # TODO: Returns the correct corresponding model. Let's start with just SVM.
    return LinearSvmModel(vocab, **kwargs)


def main(unused_argv):

    # Load the vocabulary file.
    vocab = Vocab(vocab_file, os.path.join(FLAGS.data_dir, FLAGS.input_type))

    # Load the data file.
    dataset = Dataset(FLAGS.data_dir, FLAGS.input_type, FLAGS.preprocessor,
                      FLAGS.mode, FLAGS.max_seq_len, vocab, regular_data_file,
                      debug_data_file, FLAGS.debug)

    with tf.Graph().as_default():

        # Load the model.
        print "Loading the model..."
        model = get_model(vocab, dataset)
        model.build()

        # Run the model.
        print "Running the model..."
        if FLAGS.mode == 'train':
            train(model, dataset)
        elif FLAGS.mode == 'dev':
            test(model, dataset, 'dev')
        elif FLAGS.mode == 'test':
            test(model, dataset, 'test')
        else:
            raise ValueError('Unrecognized mode: %s.' % FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()
