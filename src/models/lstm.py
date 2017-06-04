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
                shape=(self._vocab.size(), self._embedding_size),
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float64
            )

            cell = tf.contrib.rnn.LSTMCell(
                self._hidden_size,
                initializer=tf.contrib.layers.xavier_initializer()  # TODO: consider different initializers
            )

            embedded_inputs = tf.nn.embedding_lookup(embeddings, self.essay_inputs_placeholder)

            projected_embedding_inputs = tf.layers.dense(embedded_inputs,
                self._hidden_size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # TODO: consider different initializers
                name="prem_proj")

            outputs, (_, final_state) = tf.nn.dynamic_rnn(cell, embedded_inputs,
                                           sequence_length=self.essay_inputs_lengths,
                                           dtype=tf.float64)

            # Note to future self: We can use the final state as the initial state for
            # serial LSTMs, using different data sources, for example.

            # First layer
            # TODO: Add more layers, and add kernel_regularizer using l2_regularization.
            h1 = tf.layers.dense(final_state,
                self._num_classes,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.tanh,
                use_bias=False,
                name='h1')

            # Final softmax activation
            logits = tf.nn.softmax(h1, name='logits')
            preds = tf.argmax(logits, axis=1)

            return preds, logits
