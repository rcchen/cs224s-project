import csv
import json
import math
import os
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk import pos_tag_sents

from src.datasets.essays import load_essays_pos
from src.datasets.speech import load_speech_pos


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
                 pos_vocab,         # pseudo-vocabulary instance of POS tokens
                 data_file,
                 ngram_lengths,
                 pos_ngram_lengths):

        self.ngram_lengths = ngram_lengths
        self.pos_ngram_lengths = pos_ngram_lengths
        self._max_seq_len = int(max_seq_len)

        if os.path.isfile(data_file):
            with open(data_file, 'r') as f:
                self._dataframes = pickle.load(f)
            print "Loading data from the pickled file..."
        else:
            self._dataframes = self._create_dataframes(data_dir, input_type,
                                    preprocessor, max_seq_len, vocab, pos_vocab)
            print "Pickling the data object..."
            with open(data_file, "wb") as f:
                pickle.dump(self._dataframes, f)

    def _create_dataframes(self, data_dir, input_type, preprocessor, max_seq_len, vocab, pos_vocab):
        """Creates the pandas dataframes for the data."""
        df = {}

        for split in ['dev', 'train']:

            # Labels
            labels_path = ("{data_dir}/labels/{split}/labels.{split}.csv".format(
                data_dir=data_dir, split=split
            ))

            # Essays
            essays_data_path = ("{data_dir}/essays/{split}/{preprocessor}/".format(
                data_dir=data_dir, split=split, preprocessor=preprocessor
            ))

            # Speech transcriptions
            speech_transcriptions_data_path = ("{data_dir}/speech_transcriptions/{split}/{preprocessor}".format(
                data_dir=data_dir, split=split, preprocessor=preprocessor
            ))

            # i-Vectors
            ivectors_data_path = ("{data_dir}/ivectors/{split}/ivectors.json".format(
                data_dir=data_dir, split=split
            ))

            # Part of speech
            pos_data_path = ("{data_dir}/pos/{split}/{preprocessor}").format(
                data_dir=data_dir, split=split, preprocessor=preprocessor
            )

            split_features = self.extract_features(labels_path, 
                                                   essays_data_path,
                                                   speech_transcriptions_data_path,
                                                   ivectors_data_path,
                                                   pos_data_path,
                                                   max_seq_len,
                                                   vocab,
                                                   pos_vocab)

            df[split] = pd.DataFrame.from_dict(split_features)

        return df

    def pad_fn(self, seq):
        """Truncates a sequence to the max length, padding when necessary."""
        if len(seq) > self._max_seq_len:
            print "WARNING: some sequences will be truncated."
            return seq[:self._max_seq_len]
        else:
            return np.pad(seq, (0, self._max_seq_len - len(seq)), "constant")

    def extract_features(self,
                         labels_path, 
                         essays_data_path,
                         speech_transcriptions_data_path,
                         ivectors_data_path,
                         pos_data_path,
                         max_seq_len,
                         vocab,
                         pos_vocab):
        """Returns a dictionary of features, labels, and sequence lengths for the dataset."""
        df = {}
        df['essay_features'] = []
        df['essay_feature_lengths'] = []
        df['essay_pos_features'] = []
        df['speech_transcription_features'] = []
        df['speech_transcription_feature_lengths'] = []

        with open(labels_path) as labels_f:
            speaker_ids, labels = zip(*[(row['test_taker_id'], row['L1']) for
                row in csv.DictReader(labels_f)])

        # Labels
        df['labels'] = [self.CLASS_LABELS.index(l) for l in labels]

        for speaker_id in speaker_ids:

            filename = speaker_id + '.txt'

            # Essays
            with open(os.path.join(essays_data_path, filename)) as f:
                # Index all tokens
                tokens = vocab.ids_for_sentence(f.read(), self.ngram_lengths)
                df['essay_features'].append(np.array(self.pad_fn(tokens), dtype=np.int64))
                df['essay_feature_lengths'].append(min(len(tokens), max_seq_len))

            # Essay POS
            with open(os.path.join(pos_data_path, filename)) as f:
                # Index all POS n-grams
                tokens = pos_vocab.ids_for_sentence(f.read(), self.pos_ngram_lengths)
                df['essay_pos_features'].append(np.array(self.pad_fn(tokens), dtype=np.int32))

            # Speech Transcriptions
            with open(os.path.join(speech_transcriptions_data_path, filename)) as f:
                tokens = vocab.ids_for_sentence(f.read(), self.ngram_lengths)
                df['speech_transcription_features'].append(np.array(self.pad_fn(tokens), dtype=np.int64))
                df['speech_transcription_feature_lengths'].append(min(len(tokens), max_seq_len))

        # i-Vectors
        with open(ivectors_data_path) as f:
            ivector_dict = json.loads(f.read())
            df['ivectors'] = [np.array(ivector_dict[speaker_id], dtype=np.float64) for speaker_id in speaker_ids]

        return df

    def _make_batch(self, df):
        # The sequence lengths are required in order to use Tensorflow's dynamic rnn functions correctly
        return np.stack(df['essay_features']), \
               np.stack(df['essay_feature_lengths']), \
               np.stack(df['essay_pos_features']), \
               np.stack(df['speech_transcription_features']), \
               np.stack(df['speech_transcription_feature_lengths']), \
               np.stack(df['ivectors']), \
               np.stack(df['labels'])

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
