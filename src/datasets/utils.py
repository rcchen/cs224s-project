from collections import Counter
import os

import nltk

def get_data_for_path(path):
    counter = Counter()
    raw_data = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open('%s/%s' % (path, filename)) as f:
                file_tokens = []
                for line in f:
                    tokens = line.split()
                    for token in tokens:
                        counter[token] += 1
                        file_tokens.append(token)
                raw_data.append(file_tokens)
    return counter, raw_data

def get_ngrams_for_path(path, ngram_size=1):
    counter = Counter()
    raw_data = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open('%s/%s' % (path, filename)) as f:
                file_tokens = []
                for line in f:
                    tokens = line.split('')
                    for i in range(0, len(tokens) - ngram_size):
                        token = tokens[i:i+ngram_size]
                        counter[token] += 1
                        file_tokens.append(token)
                raw_data.append(file_tokens)
    return counter, raw_data

def get_pos_tags_for_data(data):
    data_tags = []
    for datum in data:
        data_tags.append([a[1] for a in nltk.pos_tag(datum)])
    return data_tags   
