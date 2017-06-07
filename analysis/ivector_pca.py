import csv
import json
import os
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector


LOG_DIR = 'logs'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
ivectors_data_path = '../var/data/ivectors/dev/ivectors.json'

with tf.Session() as sess:

    with open(metadata) as metadata_file:
        speaker_ids = [row['test_taker_id'] for row in csv.DictReader(metadata_file, delimiter='\t')]

    # i-Vectors
    with open(ivectors_data_path) as f:
        ivector_dict = json.loads(f.read())
        ivector = np.stack([np.array(ivector_dict[speaker_id], dtype=np.float64) for speaker_id in speaker_ids])

    ivector_embeddings = tf.get_variable(name='ivector_embeddings',
        initializer=tf.constant(ivector))
    saver = tf.train.Saver([ivector_embeddings])

    sess.run(ivector_embeddings.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'ivector.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = ivector_embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

