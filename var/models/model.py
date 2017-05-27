import tensorflow as tf

class NativeLanguageIdentificationModel(object):
    """Abstracts a Tensorflow graph for the NLI task."""

    def __init__(self, batch_size, max_seq_len, num_classes):
        # TODO: Initialize model with common private _variables, such as model
        # hyperparameters. Subclasses will call super(<SubClass>, self).__init__
        # to inherit this initialization method.
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._num_classes = num_classes


    def create_feed_dict(self, labels_batch, essay_inputs_batch):
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

        # TODO: Add in transcriptions, ivectors.
        # feed_dict[self.speech_transcriptions_inputs_placeholder] = speech_transcription_inputs_batch
        # feed_dict[self.ivector_inputs_placeholder] = ivector_inputs_batch

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
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name='labels')
        # Investigate why this doesn't work with batch_size?
        self.essay_inputs_placeholder = tf.placeholder(tf.int64, \
            shape=(None, self._max_seq_len), name='essay_inputs')
        # self.speech_transcriptions_inputs_placeholder = tf.placeholder(tf.int64, \
        #     shape=(None, self._max_seq_len), name='speech_transcription_inputs')
        # TODO: Replace `None` with ivector dimension size
        # self.ivector_inputs_placeholder = tf.placeholder(tf.int64, \
        #     shape=(None, None), name='ivector_inputs')


    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input
        data into logits, predictions."""
        raise NotImplementedError("Each model must implement this method.")


    def add_loss_op(self, pred, logits):
        """Adds Ops for the loss function to the computational graph."""
        loss = (
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.labels_placeholder)
            )
            + tf.cast(tf.reduce_sum(sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))), tf.float64)
        )
        tf.summary.scalar("loss", loss)
        return loss


    def add_summary_op(self):
        """Adds Ops for the summary to the computational graph."""
        return tf.summary.merge_all()


    def add_acc_op(self, preds):
        """Adds Ops for the accuracy to the computational graph."""
        return tf.contrib.metrics.accuracy(preds, self.labels_placeholder)


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
        return tf.train.AdamOptimizer().minimize(loss)


    def train_on_batch(self, sess, essay_inputs_batch, essay_inputs_len_batch, labels_batch):
        feed = self.create_feed_dict(labels_batch, essay_inputs_batch)
        acc, loss, summary = sess.run([self.acc_op, self.loss, self.summary_op], feed)
        return loss, summary


    def evaluate_on_batch(self, sess, labels_batch, essay_inputs_batch,
                          speech_transcription_inputs_batch,
                          ivector_inputs_batch):
        feed = self.create_feed_dict(labels_batch, essay_inputs_batch)
        accuracy, loss, predictions = sess.run([self.acc_op, self.loss, self.preds], feed)
        return accuracy, loss, predictions


    def build(self):
        self.add_placeholders()
        self.pred, self.logits = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred, self.logits)
        self.train_op = self.add_training_op(self.loss)
        self.acc_op = self.add_acc_op(self.pred)
        self.summary_op = self.add_summary_op()
