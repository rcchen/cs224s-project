import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.utils.dataset import Dataset

def evaluate(labels, predictions):

    labels_df = pd.read_csv(labels)
    predictions_df = pd.read_csv(predictions, header=None)

    y_true = np.concatenate(labels_df.as_matrix()[:, [3]]).ravel()
    y_pred_indices = np.concatenate(predictions_df.as_matrix()).ravel().astype(int)
    y_pred = np.array([Dataset.CLASS_LABELS[i] for i in y_pred_indices])

    print confusion_matrix(y_true, y_pred, labels=Dataset.CLASS_LABELS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, help='Labels')
    parser.add_argument('predictions', type=str, help='Predictions')
    args = parser.parse_args()
    evaluate(args.labels, args.predictions)
