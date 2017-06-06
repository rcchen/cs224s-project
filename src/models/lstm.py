import numpy as np
import tensorflow as tf

from model import NativeLanguageIdentificationModel

class LSTMModel(NativeLanguageIdentificationModel):
    """A model that trains an RNN-LSTM classifier on character ngram inputs."""

    def __init__(self, *args, **kwargs):
        super(LSTMModel, self).__init__(*args, **kwargs)


    def add_prediction_op(self):
        with tf.variable_scope('prediction'):
            # Initialize embeddings, with shape [vocab_size x hidden_size]
            # TODO: add regularizer to all trainable variables
            embeddings = tf.get_variable('embeddings',
                shape=(self._pos_vocab.size(), self._embedding_size),
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float64
            )

            cell = tf.contrib.rnn.LSTMCell(
                self._embedding_size,
                initializer=tf.contrib.layers.xavier_initializer() 
            )

            embedded_inputs = tf.nn.embedding_lookup(embeddings, self.essay_pos_inputs_placeholder)
            embedding_shape = [tf.shape(self.essay_pos_inputs_placeholder)[0], \
                               tf.shape(self.essay_pos_inputs_placeholder)[1], \
                               self._embedding_size]
            embedded_inputs = tf.reshape(embedded_inputs, shape=embedding_shape)

            outputs, (_, final_state) = tf.nn.dynamic_rnn(cell, embedded_inputs,
                                           sequence_length=self.essay_inputs_lengths,
                                           dtype=tf.float64)

            # First layer
            # TODO: Add more layers, and add kernel_regularizer using l2_regularization.
            h1 = tf.layers.dense(final_state,
                self._num_classes,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu,
                use_bias=False,
                name='h1')

            # Final softmax activation
            logits = tf.nn.softmax(h1, name='logits')
            preds = tf.argmax(logits, axis=1)

            return preds, logits
