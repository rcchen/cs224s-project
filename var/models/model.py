import tensorflow as tf

class NativeLanguageIdentificationModel(object):
     """Abstracts a Tensorflow graph for the NLI task."""

     def __init__(self, batch_size):
        # TODO: Initialize model with common private _variables, such as model
        # hyperparameters. Subclasses will call super(<SubClass>, self).__init__
        # to inherit this initialization method.
        self._batch_size = batch_size


    def create_feed_dict(self, labels_batch, essay_inputs_batch, speech_transcription_inputs_batch, ivector_inputs_batch):
        """Creates the feed_dict for one step of training.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        """
        feed_dict = {}

        feed_dict[self.labels_placeholder] = labels_batch
        feed_dict[self.essay_inputs_placeholder] = essay_inputs_batch
        feed_dict[self.speech_transcriptions_inputs_placeholder] = speech_transcription_inputs_batch
        feed_dict[self.ivector_inputs_placeholder] = ivector_inputs_batch

        return feed_dict


    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used
        as inputs by the rest of the model building and will be fed data during
        training.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size), name='labels')
        self.essay_inputs_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size, None), name='essay_inputs')
        self.speech_transcriptions_inputs_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size, None), name='speech_transcription_inputs')
        self.ivector_inputs_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size, None), name='ivector_inputs')


    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input
        data into predictions.
        Returns:
            pred: A tensor of shape (batch_size, )
            logits: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each model must implement this method.")


    def add_loss_op(self):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
            logits: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        self.loss =


    def add_summary_op(self):
        """Adds Ops for the summary to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            acc: A 0-d tensor (scalar) output
        """
        self.merged_summary_op = tf.summary.merge_all()


    def add_training_op(self):
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
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)


    def train_on_batch(self, sess, labels_batch, essay_inputs_batch,
                       speech_transcription_inputs_batch,
                       ivector_inputs_batch):
        feed = self.create_feed_dict(labels_batch, essay_inputs_batch,
                                     speech_transcription_inputs_batch,
                                     ivector_inputs_batch)
        _, loss, summary = self.run([self.optimizer, self.loss,
                                     self.merged_summary_op], feed)
        return loss, summary


    def evaluate_on_batch(self, sess, labels_batch, essay_inputs_batch,
                          speech_transcription_inputs_batch,
                          ivector_inputs_batch):
        pass


    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()
        self.add_summary_op()
