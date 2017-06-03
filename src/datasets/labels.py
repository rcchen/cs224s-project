import os

import numpy as np
import pandas as pd

from utils import get_data_for_path

# authoritatively-indexed list of language labels
CLASS_LABELS = {
    'ARA': 0,
    'CHI': 1,
    'FRE': 2,
    'GER': 3,
    'HIN': 4,
    'ITA': 5,
    'JPN': 6,
    'KOR': 7,
    'SPA': 8,
    'TEL': 9,
    'TUR': 10
}

def get_default_path(subdir):
    curdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(curdir, subdir)

def get_labels_for_path(path):
    labels = pd.read_csv(path)
    return [CLASS_LABELS[i] for i in np.concatenate(labels.as_matrix()[:, [3]]).ravel()]

def load_labels(path=get_default_path('../../var/data/labels')):
    '''
    Loads the labels dataset.

    # Arguments
        path: where the data is
    
    # Returns
        tuple of np arrays: `x_train, x_test`
    '''

    train_path = os.path.join(path, 'train/labels.train.csv')
    dev_path = os.path.join(path, 'dev/labels.dev.csv')

    return get_labels_for_path(train_path), get_labels_for_path(dev_path)    
