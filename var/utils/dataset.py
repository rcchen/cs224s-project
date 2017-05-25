import os
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np


class Dataset(object):
    def __init__(self, data_dir, vocab, debug=False):
        data_file = debug_data_file if debug else regular_data_file
        if os.path.isfile(data_file):  # todo: validate file is not corrupt
            with open(data_file, "r") as f:
                self._dataframes = pickle.load(f)[0]
            print "Loading data from the pickled file..."
        else:
            self._dataframes = self._create_dataframes(data_dir, data_file, vocab,
                                                       debug)
            print "Pickling the data object..."
            with open(data_file, "wb") as f:
                pickle.dump((self._dataframes, debug), f)


    
