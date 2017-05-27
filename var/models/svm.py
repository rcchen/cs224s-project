import numpy as np
import tensorflow as tf

from model import NativeLanguageIdentificationModel
from sklearn.feature_extraction.text import CountVectorizer

class LinearSvmModel(NativeLanguageIdentificationModel):

    def __init__(self, vocab, *args, **kwargs):
        super(LinearSvmModel, self).__init__(*args, **kwargs)
        self._vocab = vocab


    def transform_inputs_to_count_vector(self):
        """Transforms our placeholder inputs to count representations rather than
        occurrences.

        e.g. for a single example: [1, 1, 2, 0] -> [1, 2, 1]

        """
        with tf.variable_scope('inputs'):
            vocab_size = self._vocab.size()
            count_vectorizer = CountVectorizer(input='content',
                vocabulary={ str(i): i for i in range(vocab_size) },  # input is already indexed!
                lowercase=False, max_features=vocab_size)

            # BUG: Cannot iterate over variable-length tensor. Must replace this or tf will complain.
            counts = [ count_vectorizer.fit_transform(self.essay_inputs_placeholder[i, :])
                    for i in range(self._batch_size) ]

            return tf.get_variable(name='counts_matrix', dtype=tf.int64, trainable=False,
                initializer=np.stack(counts))


    def add_prediction_op(self):
        with tf.variable_scope('prediction'):
            # y = Wx
            logits = tf.layers.dense(self.transform_inputs_to_count_vector(),
                self._num_classes, use_bias=False, name='logits')
            preds = tf.argmax(logits, axis=1)
            return logits, preds


    def add_loss_op(self):
        # Override with hinge loss, instead of default cross-entropy loss.
        return tf.losses.hinge_loss(self.labels_placeholder, self.logits)