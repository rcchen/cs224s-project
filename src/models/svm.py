import tensorflow as tf

from model import NativeLanguageIdentificationModel
from src.utils.common import transform_inputs_to_count_vector


class LinearSvmModel(NativeLanguageIdentificationModel):

    def __init__(self, *args, **kwargs):
        super(LinearSvmModel, self).__init__(*args, **kwargs)


    def add_prediction_op(self):
        with tf.variable_scope('prediction'):
            # y = Wx
            encoded_inputs = tf.py_func(transform_inputs_to_count_vector,
                [self._vocab.size(), self.essay_inputs_placeholder], tf.float64)
            encoded_inputs.set_shape((None, self._vocab.size()))
            logits = tf.layers.dense(encoded_inputs, self._num_classes, use_bias=False, name='logits')
            preds = tf.argmax(logits, axis=1)
            return preds, logits


    def add_loss_op(self, pred, logits):
        # Override with hinge loss, instead of default cross-entropy loss.
        labels = tf.one_hot(indices=self.labels_placeholder, depth=self._num_classes)
        loss = tf.reduce_mean(tf.losses.hinge_loss(labels, logits))
        tf.summary.scalar('loss', loss)
        return loss
