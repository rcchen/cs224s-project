import numpy as np
import tensorflow as tf

from model import NativeLanguageIdentificationModel
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.estimators import svm
from tensorflow.python.framework import constant_op

# disable tensorflow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

class LinearSvmModel(NativeLanguageIdentificationModel):
    """Model using classifier at tf.contrib.learn.SVM."""

    def __init__(self, vocab, *args, **kwargs):
        super(LinearSvmModel, self).__init__(*args, **kwargs)
        self._vocab = vocab
        self._svm_classifier = svm.SVM(feature_columns=self.get_feature_columns(),
                                       example_id_column='example_id',
                                       l1_regularization=0.0,  # TODO: use config for this
                                       l2_regularization=0.0)

    def get_feature_columns(self):
        return [feature_column.real_valued_column(str(vocab_index))
                    for vocab_index in range(self._vocab.size())]

    def add_prediction_op(self):
        # Predicts the labels.
        # self.preds = self._svm_classifier.predict(input_fn=self.input_fn)
        pass

    def add_training_op(self):
        pass

    def add_loss_op(self):
        # Evaluates our classifier on the batch data.
        pass

    def train_on_batch(self, sess, essay_inputs_batch, essay_inputs_len_batch, labels_batch):

        def input_fn():
            dt_matrix = np.zeros((self._batch_size, self._vocab.size()))

            # Transform the text input into a matrix of counts
            for doc_index, input_length in zip(range(self._batch_size), essay_inputs_len_batch):
                for input_index in range(input_length):
                    word_index = essay_inputs_batch[doc_index, input_index]
                    dt_matrix[doc_index, word_index] += 1

            n_samples, n_features = dt_matrix.shape

            # Some data sanity checks
            assert n_samples == len(labels_batch)
            assert n_features == self._vocab.size()

            data = { str(i) : constant_op.constant(dt_matrix[:, i], dtype=np.int64) for i in range(n_features)}
            data['example_id'] = constant_op.constant([ str(example_id) for example_id in range(n_samples) ])
            return data, constant_op.constant(labels_batch)

        self._svm_classifier.fit(input_fn=input_fn, steps=1)
        self._metrics = self._svm_classifier.evaluate(input_fn=input_fn, steps=1)

        feed = self.create_feed_dict(labels_batch, essay_inputs_batch, None, None)
        summary = sess.run(self.merged_summary_op, feed)
        return self._metrics['loss'], summary

    def build(self):
        self.add_placeholders()
