import numpy as np
import tensorflow as tf

from model import NativeLanguageIdentificationModel

class RNNModel(NativeLanguageIdentificationModel):
    """A model that trains an RNN classifier on POS ngram inputs."""

    def __init__(self, *args, **kwargs):
        super(RNNModel, self).__init__(*args, **kwargs)


    def add_prediction_op(self):
        with tf.variable_scope('prediction'):
            # Initialize embeddings, with shape [pos_tags x hidden_size]
            # TODO: add regularizer to all trainable variables
            embeddings = tf.get_variable('embeddings',
                shape=(45, self._embedding_size),
                initializer=tf.contrib.layers.xavier_initializer(),  # TODO: consider different initializers
                dtype=tf.float64
            )

    	    cell = tf.contrib.rnn.BasicRNNCell(num_units=self._hidden_size)

            embedded_inputs = tf.nn.embedding_lookup(tf.constant(self._embedding_matrix),
                self.essay_pos_inputs_placeholder)
            embedding_shape = [tf.shape(self.essay_pos_inputs_placeholder)[0], \
                               tf.shape(self.essay_pos_inputs_placeholder)[1], \
                               self._embedding_size]
            embedded_inputs = tf.reshape(embedded_inputs, shape=embedding_shape)

            # TODO: Investigate how to only train on unseen, keeping the GloVe vectors intact.

            projected_embedding_inputs = tf.layers.dense(embedded_inputs,
                self._embedding_size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="prem_proj")

            outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                inputs=embedded_inputs,
                sequence_length=self.essay_inputs_lengths,
                dtype=tf.float64)

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
