#!/usr/bin/env python
# -*- coding: utf-8 -*-
# source: https://github.com/DeNeutoy/Decomposable_Attn/blob/master/Vocab.py

import os, glob, json
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize

# Authoritatively indexed list of POS from Penn Treebank
POS_TAGS = ['CC', 'CD' 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD',
'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

class Vocab(object):

    def __init__(self, vocab_file, dataset_path, ngram_lengths, max_vocab_size=10000):
        self.token_id = {}
        self.id_token = {}
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.seq = 2
        self.vocab_file = vocab_file

        if os.path.isfile(vocab_file) and os.path.getsize(vocab_file) > 0:
            self.load_vocab_from_file(vocab_file)
        elif dataset_path:
            self.create_vocab(dataset_path, vocab_file, max_vocab_size, ngram_lengths)
            self.load_vocab_from_file(vocab_file)
        else:
            raise Exception("must provide either an already constructed vocab file, or a dataset to build it from.")

    def load_vocab_from_file(self, vocab_file):
        print("loading vocab from {}".format(vocab_file))

        for line in open(vocab_file, "r"):
            token, idx = line.strip().split("\t")
            idx = int(idx)
            assert token not in self.token_id, "dup entry for token [%s]" % token
            assert idx not in self.id_token, "dup entry for idx [%s]" % idx
            if idx == 0:
                assert token == "PAD", "expect id 0 to be [PAD] not [%s]" % token
            if idx == 1:
                assert token == "UNK", "expect id 1 to be [UNK] not [%s]" % token
            self.token_id[token] = idx
            self.id_token[idx] = token

    def create_vocab(self, dataset_path, vocab_file, max_vocab_size, ngram_lengths):

        print("generating vocab from dataset at {}".format(dataset_path))
        all_words = []
        for filename in glob.glob(os.path.join(dataset_path, '*', 'tokenized',
                                               '*.txt')):
            with open(filename, 'r') as f:
                for ngram_length in ngram_lengths:
                    all_words += self.tokenize_ngrams(f.read().lower(), ngram_length)

        counter = Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x : (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        words = ["PAD"] + ["UNK"] + list(words)
        word_to_id = dict(zip(words[:max_vocab_size], range(max_vocab_size)))

        with open(vocab_file, "w") as file:
            for word, id in word_to_id.items():
                file.write("{}\t{}\n".format(word.encode('utf-8') , id))

        print("vocab of size {} written to {}, with PAD token == 0, UNK token == 1".format(max_vocab_size,vocab_file))


    def size(self):
        return len(self.token_id) + 2  # +1 for UNK & PAD

    def id_for_token(self, token):
        if token in self.token_id:
            return self.token_id[token]
        return self.UNK_ID

    def ids_for_tokens(self, tokens):
        return [self.id_for_token(t) for t in tokens]

    def ids_for_sentence(self, sentence, ngram_lengths):
        ids = []
        for ngram_length in ngram_lengths:
            ids += self.ids_for_tokens(self.tokenize_ngrams(sentence.lower(), ngram_length))
        return ids

    def token_for_id(self, id):

        if id in self.id_token:
            return self.id_token[id]

        else:
            print("ID not in vocab, returning <UNK>")
            return self.UNK_ID

    def tokens_for_ids(self, ids):
        return [self.token_for_id(x) for x in ids]

    def has_token(self, token):
        return token in self.token_id

    @staticmethod
    def tokenize_ngrams(text, n):
        if n == 0:  # special case: tokenize by word
            return word_tokenize(text.decode('utf-8'))
        if len(text) < n:
            stripped_text = text.strip()
            if len(stripped_text) != 0:
                return [stripped_text]
            else:
                return []
        sentences = [line.strip() for line in text.split('\n')]
        tokens = []
        for sentence in sentences:
            tokens += [sentence[i:i+n] for i in range(len(sentence) - n)]
        return tokens
