import tensorflow as tf

from .model import NativeLanguageIdentificationModel

class LinearSvmModel(NativeLanguageIdentificationModel):
    """Model using classifier at tf.contrib.learn.SVM."""

    
