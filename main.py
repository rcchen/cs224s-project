from datetime import datetime
import os

import tensorflow as tf
import numpy as np

from src.models import *
from src.utils.dataset import Dataset
from src.utils.progbar import Progbar
from src.utils.vocab import Vocab
from src.utils.glove import get_glove_vectors

flags = tf.app.flags

# Running conditions
flags.DEFINE_string('name', '', 'The name of the model.')
flags.DEFINE_string('model', 'nn', 'The name of the model to run.')
flags.DEFINE_string('mode', 'train', 'Running mode: either "dev", "train", or "test"')
flags.DEFINE_string('data_dir', 'var/data', 'The directory containing data files.')
flags.DEFINE_string('output_dir', 'output', 'The directory for output to go.')
flags.DEFINE_string('glove_file', 'glove.twitter.27B.25d.txt', 'The name of the glove file. Must match with embedding size.')
flags.DEFINE_string('glove_saved_file', 'glove.twitter.27B.25d.npy', 'The name of the saved numpy glove vectors for the vocab.')

# Data parameters
flags.DEFINE_string('input_type', 'essays', 'Input data feature type: either "essays", \
                    "speech_transcriptions", "ivectors", or \
                    "speech_transcriptions+ivectors" ')
flags.DEFINE_string('preprocessor', 'tokenized', 'Name of directory with processed essay files.')
flags.DEFINE_string('ngram_lengths', '0,3,4,5', 'Comma-separated list of n-gram sizes to use as features.')
flags.DEFINE_string('pos_ngram_lengths', '0', 'Comma-separated list of POS n-gram sizes to use as features.')

flags.DEFINE_integer('max_seq_len', 1e4, 'Max number of words in an example.')
flags.DEFINE_integer('batch_size', 50, 'Number of examples to run in a batch.')
flags.DEFINE_integer('num_epochs', 25, 'Number of epochs to train for.')
flags.DEFINE_integer('embedding_size', 30, 'Size of trainable embeddings, applicable for char-gram embedding models.')
flags.DEFINE_integer('hidden_size', 250, 'Number of cells in a neural network layer.')

# Training and testing
flags.DEFINE_string('train_split', 'train', 'Split to train the model on.')
flags.DEFINE_string('dev_split', 'dev', 'Split to evaluate the model on.')
flags.DEFINE_boolean('save', True, 'Whether to save the model.')

# Model hyperparamters
flags.DEFINE_float('l2_reg', 1e-4, 'The L2 regularization coefficient')

# TODO: add dropout to our models.
flags.DEFINE_float('dropout_rate', 0.15, 'How many units to eliminate during training, applicable to models using dropout.')

# TODO: use or delete this.
flags.DEFINE_string('debug', False, 'Run on debug mode, using a smaller data set.')

FLAGS = flags.FLAGS

# File paths
vocab_dir = os.path.join(FLAGS.output_dir, 'vocabs')
vocab_file = os.path.join(vocab_dir, 'ngrams-%s.txt' % FLAGS.ngram_lengths)
speech_vocab_file = os.path.join(vocab_dir, 'speech-vocab.txt')
pos_vocab_file = os.path.join(vocab_dir, 'pos-ngrams-%s.txt' % FLAGS.pos_ngram_lengths)
glove_file = os.path.join(FLAGS.data_dir, 'glove', FLAGS.glove_file) 
glove_saved_file = os.path.join(FLAGS.data_dir, 'glove', FLAGS.glove_saved_file) 
pickle_dir = os.path.join(FLAGS.output_dir, 'pickles')
pickle_file = os.path.join(pickle_dir, '%s_%s_ngrams-%s_data.pkl' % 
    (FLAGS.input_type, FLAGS.preprocessor, FLAGS.ngram_lengths))
summary_dir = os.path.join(FLAGS.output_dir, 'tensorboard')

# TODO: Customize path so that we can save and load 1< checkpoints for a single model, w/ different hyperparameters.
checkpoint_dir = os.path.join(FLAGS.output_dir, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, '%s_model.ckpt' % FLAGS.model)
predictions_dir = os.path.join(FLAGS.output_dir, 'predictions')

# Ensure that all necessary output directories exist.
output_dirs = [FLAGS.output_dir, vocab_dir, pickle_dir, checkpoint_dir, predictions_dir]
for d in output_dirs:
    if not os.path.isdir(d):
        os.mkdir(d)


def run_train_epoch(sess, model, dataset, epoch_num, summary_writer):
    print '='*79
    print 'Epoch: %s' % (epoch_num + 1)
    prog = Progbar(target = dataset.split_num_batches(FLAGS.train_split, FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_shuffled_iterator(FLAGS.train_split, FLAGS.batch_size)):
        loss, summary = model.train_on_batch(sess, *batch)
        summary_writer.add_summary(summary, global_step=epoch_num * dataset.split_size(FLAGS.train_split) + i)
        prog.update(i + 1, [('train loss', loss)])
    print '='*79


def run_eval_epoch(sess, model, dataset):
    batch_sizes = []
    accuracies = []
    preds = []
    speaker_ids = []

    print '-'*79
    prog = Progbar(target=dataset.split_num_batches(FLAGS.dev_split, FLAGS.batch_size))
    for i, batch in enumerate(dataset.get_iterator(FLAGS.dev_split, FLAGS.batch_size)):
        acc, loss, pred = model.evaluate_on_batch(sess, *batch)
        prog.update(i + 1, [('loss', loss)])

        batch_sizes.append(batch[0].shape[0])
        accuracies.append(acc)
        preds.append(pred)
        speaker_ids.append(batch[-1]) # speaker_id

    accuracy = np.average(accuracies, weights=batch_sizes)
    print 'Accuracy: %s' % accuracy
    print '-'*79
    return accuracy, np.concatenate(preds), np.concatenate(speaker_ids)


def train(model, dataset):
    if FLAGS.save:
        saver = tf.train.Saver(max_to_keep=1)

    # Allow GPU memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        best_accuracy = 0
        for epoch in range(FLAGS.num_epochs):
            run_train_epoch(sess, model, dataset, epoch, summary_writer)
            dev_accuracy, predictions, speaker_ids = run_eval_epoch(sess, model, dataset)
            if dev_accuracy > best_accuracy:
                saver.save(sess, checkpoint_path)
                best_accuracy = dev_accuracy

                # Save predictions and corresponding speaker IDs
                print 'Better model found. Saving predictions to %s/%s.csv' % (predictions_dir, FLAGS.name)
                np.savetxt("%s/%s.csv" % (predictions_dir, FLAGS.name), predictions, delimiter=",")
                np.savez("%s/%s-predictions-and-speakers.npz" % (predictions_dir, FLAGS.name), 
                    predictions=predictions, speaker_ids=speaker_ids)


def test(model, dataset):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, checkpoint_path)
        _, predictions = run_eval_epoch(sess, model, dataset)
        timestamp = datetime.utcnow().strftime("%s")
        np.savetxt("%s/%s/%s.csv" % (predictions_dir, FLAGS.mode, timestamp), predictions, delimiter=",")       

def get_model(vocab, pos_vocab, speech_vocab, dataset):
    kwargs = {
        'batch_size': FLAGS.batch_size,
        'max_seq_len': FLAGS.max_seq_len,
        'num_classes': len(dataset.CLASS_LABELS),
        'l2_reg': FLAGS.l2_reg,
        'vocab': vocab,
        'pos_vocab': pos_vocab,
        'speech_vocab': speech_vocab,
        'embedding_size': FLAGS.embedding_size,
        'hidden_size': FLAGS.hidden_size
    }

    if FLAGS.model == 'baseline':
        return LinearSvmModel(**kwargs)
    elif FLAGS.model == 'rnn':
        return RNNModel(**kwargs)
    elif FLAGS.model == 'lstm':
        return LSTMModel(**kwargs)
    elif FLAGS.model == 'nn':
        return MultilayerNeuralNetModel(**kwargs)
    else:
        raise ValueError("Unrecognized model: %s." % FLAGS.model)


def main(unused_argv):

    # Load the vocabulary file.
    ngram_lengths = [int(i) for i in FLAGS.ngram_lengths.split(',')]
    pos_ngram_lengths = [int(i) for i in FLAGS.pos_ngram_lengths.split(',')]

    vocab = Vocab(vocab_file, os.path.join(FLAGS.data_dir, 'essays'), ngram_lengths)
    pos_vocab = Vocab(pos_vocab_file, os.path.join(FLAGS.data_dir, 'pos'), pos_ngram_lengths)
    # Speech only uses single tokens as features. Not char n-grams nor POS.
    speech_vocab = Vocab(speech_vocab_file, os.path.join(FLAGS.data_dir, 'speech_transcriptions'), [0])

    # Load the data file.
    dataset = Dataset(FLAGS.data_dir, FLAGS.input_type, FLAGS.preprocessor,
                      FLAGS.max_seq_len, vocab, speech_vocab, pos_vocab, pickle_file,
                      ngram_lengths, pos_ngram_lengths)

    with tf.Graph().as_default():

        # Load the model.
        print "Loading the model..."
        model = get_model(vocab, pos_vocab, speech_vocab, dataset)
        model.build()

        # Run the model.
        print "Running the model..."
        if FLAGS.mode == 'train':
            train(model, dataset)
        elif FLAGS.mode == 'dev':
            test(model, dataset)
        elif FLAGS.mode == 'test':
            test(model, dataset)
        else:
            raise ValueError('Unrecognized mode: %s.' % FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()
