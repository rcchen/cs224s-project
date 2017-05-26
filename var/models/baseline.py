import tensorflow as tf

from model import NativeLanguageIdentificationModel
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.estimators import svm


class LinearSvmModel(NativeLanguageIdentificationModel):
    """Model using classifier at tf.contrib.learn.SVM."""

    # BASELINE DETAILS
    # Baseline uses tf.contrib.learn.SVM, a well-defined classifier. It takes in
    # FeatureColumns, and we just call `fit` and `evaluate` on the classifier
    # instance.

    # TRAINING AND EVALUATING SVM
    # `fit` represents an instance of training, and `evaluate` gives us both the
    # loss and accuracy metrics. It uses SDCAOptimizer by default.

    # DATA VECTORIZATION
    # We will also need to implement something like a CountVectorizer. Right now
    # our dataset wraps around a pandas DataFrame containing vocab indices of
    # tokens from our dataset. While this will help us train our models on
    # sequence-aware neural nets, SVM will need to use token counts instead.

    # More information about SVM
    # Documentation: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/SVM
    # Example usage: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/learn/python/learn/estimators/svm_test.py


    def __init__(self, vocab, *args, **kwargs):
        super(LinearSvmModel, self).__init__(*args, **kwargs)
        self._vocab = vocab
        self._svm_classifier = svm.SVM(feature_columns=self.get_feature_columns(),
                                       example_id_column='example_id',
                                       l1_regularization=0.0,  # TODO: use config for this
                                       l2_regularization=0.0)

    def get_feature_columns(self):
        return [sparse_column_with_integerized_feature(vocab_index,
                    bucket_size=self._max_seq_len)
                    for vocab_index in range(self._vocab.size)]

    def input_fn(self):
        vectorizer = CountVectorizer(input='content',
                                     vocabulary=self._vocab.token_id,
                                     dtype=np.int64)
        dt_matrix = vectorizer.fit_transform(self.essay_inputs_placeholder)
        n_samples, n_features = dt_matrix.shape

        # Some data sanity checks
        assert n_samples == len(self.labels_placeholder)
        assert n_features == self._vocab.size()

        data = { i : dt_matrix[:, i] for i in range(n_features)}
        data['example_id'] = range(n_samples)
        return data, self.labels_placeholder

    def add_prediction_op(self):
        # Predicts the labels.
        return self._svm_classifier.predict(input_fn=self.input_fn)

    def add_loss_op(self):
        # Evaluates our classifier on the batch data.
        self._metrics = self._svm_classifier.evaluate(input_fn=input_fn, steps=1)
        return self._metrics['loss']

    def add_training_op(self):
        # Runs once through the batch data.
        self._svm_classifier.fit(input_fn=input_fn, steps=1)
