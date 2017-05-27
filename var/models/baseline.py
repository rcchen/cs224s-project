import numpy as np
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.estimators import svm
from tensorflow.python.framework import constant_op

class LinearSvmModel(object):
    """Model using classifier at tf.contrib.learn.SVM."""

    def __init__(self, vocab, *args, **kwargs):
        super(LinearSvmModel, self).__init__(*args, **kwargs)
        self._vocab = vocab
        self._svm_classifier = svm.SVM(feature_columns=self.get_feature_columns(),
                                       example_id_column='example_id',
                                       l1_regularization=0.0,  # TODO: use config for this
                                       l2_regularization=0.0)


    def build(self):
        #
        # Load the training and test features and labels
        #
        training_and_test_data = get_features_and_labels(train_partition_name,
                                                         test_partition_name,
                                                         feature_file_train,
                                                         feature_file_test,
                                                         baseline=BASELINE,
                                                         preprocessor=preprocessor,
                                                         vectorizer=vectorizer,
                                                         transformer=transformer)

        train_matrix, encoded_train_labels, original_training_labels = training_and_test_data[0]
        test_matrix, encoded_test_labels, original_test_labels = training_and_test_data[1]
