import numpy as np
import tensorflow as tf

from model import NativeLanguageIdentificationModel
from src.utils.common import transform_inputs_to_count_vector

class MultilayerNeuralNetModel(NativeLanguageIdentificationModel):

    def __init__(self, vocab, hidden_size, *args, **kwargs):
        super(MultilayerNeuralNetModel, self).__init__(*args, **kwargs)
        self._vocab = vocab
        self._hidden_size = hidden_size


    def add_prediction_op(self):
        """Runs the inputs through a multilayer NN."""
        with tf.variable_scope('prediction'):
            encoded_inputs = tf.py_func(transform_inputs_to_count_vector,
                [self._vocab.size(), self.essay_inputs_placeholder], tf.float64)
            encoded_inputs.set_shape((None, self._vocab.size()))

            # TODO: make initializer, regularizer configurable as flags.
            h1 = tf.layers.dense(encoded_inputs, self._hidden_size,
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
