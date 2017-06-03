import numpy as np
import tensorflow as tf

from model import NativeLanguageIdentificationModel
from src.utils.common import transform_inputs_to_count_vector

class MultilayerNeuralNetModel(NativeLanguageIdentificationModel):

    def __init__(self, *args, **kwargs):
        super(MultilayerNeuralNetModel, self).__init__(*args, **kwargs)


    def add_prediction_op(self):
        """Runs the inputs through a multilayer NN."""
        with tf.variable_scope('prediction'):

            #embeddings = tf.get_variable('embeddings',
            #    shape=(self._vocab.size(), self._embedding_size),
            #    initializer=tf.contrib.layers.xavier_initializer(),  # TODO: consider different initializers
            #    dtype=tf.float64
            #)

	    embeddings = tf.get_variable('embedding', initializer=tf.constant(self._embedding_matrix), dtype=tf.float64,)

            embedded_inputs = tf.reduce_sum(
                tf.nn.embedding_lookup(embeddings, self.essay_inputs_placeholder),
                axis=1
            )

            # TODO: make initializer, regularizer configurable as flags.
            h1 = tf.layers.dense(embedded_inputs, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 activation=tf.tanh, name='h1')
            
            h2 = tf.layers.dense(h1, self._hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 activation=tf.tanh, name='h2')

            logits = tf.layers.dense(h2, self._num_classes,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                     name='logits')

            preds = tf.argmax(logits, axis=1)
            return preds, logits
