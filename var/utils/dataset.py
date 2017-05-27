import csv
import math
import os
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np


class Dataset(object):

    # authoritatively-indexed list of language labels
    CLASS_LABELS = [
        'ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'
    ]

    def __init__(self,
                 data_dir,          # directory containing the data
                 input_type,        # e.g. 'essays', 'speech_transcriptions'
                 preprocessor,      # e.g. 'tokenized'
                 max_seq_len,       # max length of a sequence to ingest
                 vocab,             # Vocabulary instance
                 regular_data_file,
                 debug_data_file,
                 debug):

        data_file = debug_data_file if debug else regular_data_file
        if os.path.isfile(data_file):
            with open(data_file, 'r') as f:
                self._dataframes = pickle.load(f)
            print "Loading data from the pickled file..."
        else:
            self._dataframes = self._create_dataframes(data_dir, input_type,
                                    preprocessor, max_seq_len, vocab)
            print "Pickling the data object..."
            with open(data_file, "wb") as f:
                pickle.dump(self._dataframes, f)

    def _create_dataframes(self, data_dir, input_type, preprocessor, max_seq_len, vocab):
        """Creates the pandas dataframes for the data."""
        df = {}
        for split in ['dev', 'train']:

            labels_path = ("{data_dir}/labels/{split}/labels.{split}.csv".format(
                data_dir=data_dir, split=split
            ))
            data_path = ("{data_dir}/{input_type}/{split}/{preprocessor}/".format(
                data_dir=data_dir, input_type=input_type, split=split,
                preprocessor=preprocessor,
            ))

            with open(labels_path) as labels_f:
                data_files, labels = \
                    zip(*[(os.path.join(data_path, row['test_taker_id'] + '.txt'),
                            row['L1']) for row in csv.DictReader(labels_f)])
                df[split] = pd.DataFrame(self.extract_features(data_files,
                                         labels, vocab, max_seq_len))

        return df

    @staticmethod
    def pad_fn(seq, max_seq_len):
        """Truncates a sequence to the max length, padding when necessary."""
        if len(seq) > max_seq_len:
            print "WARNING: some sequences will be truncated."
            return seq[:max_seq_len]
        else:
            return np.pad(seq, (0, max_seq_len - len(seq)), "constant")

    def extract_features(self, file_list, labels, vocab, max_seq_len):
        """Returns a dictionary of features, labels, and sequence lengths for the dataset."""
        df = {}
        df['labels'] = np.array([self.CLASS_LABELS.index(l) for l in labels], dtype=np.int64)
        df['features'] = []
        df['lengths'] = []
        for filename in file_list:
            with open(filename) as f:
                tokens = vocab.ids_for_sentence(f.read())
                df['features'].append(np.array(self.pad_fn(tokens, max_seq_len), dtype=np.int64))
                df['lengths'].append(len(tokens))

        # Data consistency check
        assert len(df['labels']) == len(df['features']) == len(df['lengths'])

        return df

    def _make_batch(self, df):
        # The sequence lengths are required in order to use Tensorflow's dynamic rnn functions correctly
        return np.stack(df['features']), np.stack(df['lengths']), np.stack(df['labels'])

    def _make_iterator(self, df, batch_size):
        total_examples = len(df)
        examples_read = 0
        while examples_read + batch_size <= total_examples:
            yield self._make_batch(df[examples_read:examples_read + batch_size])
            examples_read += batch_size
        if examples_read < total_examples:  # there are still examples left to return
            yield self._make_batch(df[examples_read:])

    def get_iterator(self, split, batch_size):
        return self._make_iterator(self._dataframes[split], batch_size)

    def get_shuffled_iterator(self, split, batch_size):
        df = self._dataframes[split]
        return self._make_iterator(df.sample(len(df)), batch_size)

    def split_size(self, split):
        return len(self._dataframes[split])

    def split_num_batches(self, split, batch_size):
        return int(math.ceil(float(len(self._dataframes[split])) / batch_size))
