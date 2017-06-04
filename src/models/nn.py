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

            embeddings = tf.get_variable('embedding', initializer=tf.constant(self._embedding_matrix), dtype=tf.float64)

            # ESSAY INPUTS

            embedded_essay_inputs = tf.reduce_sum(
                tf.nn.embedding_lookup(embeddings, self.essay_inputs_placeholder),
                axis=1
            )

            es_h1 = tf.layers.dense(embedded_essay_inputs, self._hidden_size, use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 activation=tf.tanh, name='es_h1')
            
            es_h2 = tf.layers.dense(es_h1, self._hidden_size, use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 activation=tf.tanh, name='es_h2')

            # SPEECH TRANSCRIPTION INPUTS
            embedded_speech_inputs = tf.reduce_sum(
                tf.nn.embedding_lookup(embeddings, self.speech_transcriptions_inputs_placeholder),
                axis=1
            )

            sp_h1 = tf.layers.dense(embedded_essay_inputs, self._hidden_size, use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 activation=tf.tanh, name='sp_h1')

            sp_h2 = tf.layers.dense(embedded_essay_inputs, self._hidden_size, use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 activation=tf.tanh, name='sp_h2')

            # IVECTOR INPUTS: originally 800d
            ivector = self.ivector_inputs_placeholde

            iv_h1 = tf.layers.dense(embedded_essay_inputs, self._hidden_size, use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 activation=tf.tanh, name='iv_h1')

            # COMBINE ALL INPUTS
            total_features = tf.concat([es_h2, sp_h2, iv_h1], axis=1)

            logits = tf.layers.dense(total_features, self._num_classes, use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                     name='logits')

            preds = tf.argmax(logits, axis=1)
            return preds, logits
