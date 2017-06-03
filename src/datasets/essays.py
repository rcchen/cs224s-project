import os

import nltk
import numpy as np

from utils import get_data_for_path, get_ngrams_for_path, get_pos_tags_for_data

def get_default_path(subdir):
    curdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(curdir, subdir)

def load_essays_data(path=get_default_path('../../var/data/essays'),
                     num_words=None,
                     start_char=1,
                     oov_char=2,
                     ngram_length=0):
    '''
    Loads the essays dataset.

    # Arguments
        path: where the data is
        num_words: max number of words to include
        start_char: character to mark start sequence
        oov_char: words cut out because of num_words
    
    # Returns
        tuple of np arrays: `x_train, x_test`
    '''

    train_path = os.path.join(path, 'train/tokenized')
    dev_path = os.path.join(path, 'dev/tokenized')

    if not ngram_length == 0:
        counter, raw_train_data = get_ngrams_for_path(train_path)
        _, raw_dev_data = get_ngrams_for_path(dev_path)
    else:
        counter, raw_train_data = get_data_for_path(train_path)
        _, raw_dev_data = get_data_for_path(dev_path)

    vocab_index = {}
    current_char = 3
    for w in counter.most_common(num_words - 3):
        vocab_index[w[0]] = current_char
        current_char += 1

    x_train = []
    for raw_tokens in raw_train_data:
        substituted_tokens = [start_char]
        for raw_token in raw_tokens:
            if raw_token in vocab_index:
                substituted_tokens.append(vocab_index[raw_token])
            else:
                substituted_tokens.append(oov_char)
        x_train.append(substituted_tokens)

    x_dev = []
    for raw_tokens in raw_dev_data:
        substituted_tokens = [start_char]
        for raw_token in raw_tokens:
            if raw_token in vocab_index:
                substituted_tokens.append(vocab_index[raw_token])
        x_dev.append(substituted_tokens)

    return np.array(x_train), np.array(x_dev)

def load_essays_pos(path=get_default_path('../../var/data/essays')):
    cache_file = get_default_path('../../output/essays.pos.npz')

    if os.path.isfile(cache_file):
        data = np.load(cache_file)
        x_train = data["x_train"]
        x_dev = data["x_dev"]
        return x_train, x_dev

    else:
        train_path = os.path.join(path, 'train/tokenized')
        dev_path = os.path.join(path, 'dev/tokenized')

        _, raw_train_data = get_data_for_path(train_path)
        _, raw_dev_data = get_data_for_path(dev_path)

        x_train = get_pos_tags_for_data(raw_train_data)
        x_dev = get_pos_tags_for_data(raw_dev_data)

        np.savez(cache_file, x_train=x_train, x_dev=x_dev)

        return np.array(x_train), np.array(x_dev)
