import tensorflow as tf

class NativeLanguageIdentificationModel(object):
     """Abstracts a Tensorflow graph for the NLI task."""

     def __init__(self):
        # TODO: Initialize model with common private _variables, such as model
        # hyperparameters. Subclasses will call super(<SubClass>, self).__init__
        # to inherit this initialization method.
        raise NotImplementedError


    def create_feed_dict(self):
        """Creates the feed_dict for one step of training.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        """
        raise NotImplementedError


    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used
        as inputs by the rest of the model building and will be fed data during
        training.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError


    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input
        data into predictions.
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
            logits: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each model must implement this method.")


    def add_loss_op(self, pred, logits):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
            logits: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError


    def add_accuracy_op(self, pred):
        """Adds Ops for the accuracy to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            acc: A 0-d tensor (scalar) output
        """
        raise NotImplementedError


    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable
        variables. The Op returned by this function is what must be passed to
        the sess.run() to train the model. See
        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer
        for more information.
        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """
        raise NotImplementedError


    def build(self):
        self.add_placeholders()
        self.pred, self.logits = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred, self.logits)
        self.train_op = self.add_training_op(self.loss)
        self.acc_op = self.add_acc_op(self.pred)
