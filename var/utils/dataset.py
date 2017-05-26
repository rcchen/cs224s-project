import csv
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
                 mode,              # e.g. 'train', 'dev'
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
                                    preprocessor, mode, max_seq_len, vocab)
            print "Pickling the data object..."
            with open(data_file, "wb") as f:
                pickle.dump(self._dataframes, f)


    def _create_dataframes(self, data_dir, input_type, preprocessor, mode, max_seq_len, vocab):
        """Creates the pandas dataframes for the data."""
        labels_path = ("{data_dir}/labels/{mode}/labels.{mode}.csv".format(
            data_dir=data_dir, mode=mode
        ))
        data_path = ("{data_dir}/{input_type}/{mode}/{preprocessor}/".format(
            data_dir=data_dir, input_type=input_type, mode=mode,
            preprocessor=preprocessor
        ))

        with open(labels_path) as labels_f:
            data_files, labels = \
                zip(*[(os.path.join(data_path, row['test_taker_id'] + '.txt'),
                        row['L1']) for row in csv.DictReader(labels_f)])

        df = self.extract_features(data_files, labels, vocab, max_seq_len)
        return pd.DataFrame(df)


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
                df['features'].append(self.pad_fn(tokens, max_seq_len))
                df['lengths'].append(len(tokens))
        return df
