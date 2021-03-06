import argparse

import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support

from src.utils.dataset import Dataset

def evaluate(labels, predictions):

    labels_df = pd.read_csv(labels)
    predictions_df = pd.read_csv(predictions, header=None)

    y_true = np.concatenate(labels_df.as_matrix()[:, [3]]).ravel()
    y_pred_indices = np.concatenate(predictions_df.as_matrix()).ravel().astype(int)
    y_pred = np.array([Dataset.CLASS_LABELS[i] for i in y_pred_indices])

    # cm = ConfusionMatrix(y_true, y_pred)
    # cm.print_stats()
    stats = np.transpose(precision_recall_fscore_support(y_true, y_pred))

    for col in stats:
        print ",".join([str(a) for a in col.tolist()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, help='Labels')
    parser.add_argument('predictions', type=str, help='Predictions')
    args = parser.parse_args()
    evaluate(args.labels, args.predictions)
