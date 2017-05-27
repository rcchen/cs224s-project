import numpy as np
import tensorflow as tf

from model import NativeLanguageIdentificationModel
from sklearn.feature_extraction.text import CountVectorizer

def transform_inputs_to_count_vector(vocab_size, essay_inputs):
    """Transforms our placeholder inputs to count representations rather than
    occurrences.

    e.g. for a single example: [1, 1, 2, 0] -> [1, 2, 1]

    """
    count_vectorizer = CountVectorizer(input='content',
        vocabulary={ str(i): i for i in range(vocab_size) },  # input is already indexed!
        lowercase=False, max_features=vocab_size)

    # BUG: Cannot iterate over variable-length tensor. Must replace this or tf will complain.
    counts = [ count_vectorizer.fit_transform(essay_inputs[i, :])
            for i in range(essay_inputs.shape[0]) ]

    return np.stack(counts)

class LinearSvmModel(NativeLanguageIdentificationModel):

    def __init__(self, vocab, *args, **kwargs):
        super(LinearSvmModel, self).__init__(*args, **kwargs)
        self._vocab = vocab


    def add_prediction_op(self):
        with tf.variable_scope('prediction'):
            # y = Wx
            encoded_inputs = tf.py_func(transform_inputs_to_count_vector,
                [self._vocab.size(), self.essay_inputs_placeholder], tf.int32)
            encoded_inputs.set_shape((None, self._vocab.size()))
            logits = tf.layers.dense(encoded_inputs, self._num_classes, use_bias=False, name='logits')
            preds = tf.argmax(logits, axis=1)
            return logits, preds


    def add_loss_op(self, pred, logits):
        # Override with hinge loss, instead of default cross-entropy loss.
        return tf.losses.hinge_loss(self.labels_placeholder, self.logits)