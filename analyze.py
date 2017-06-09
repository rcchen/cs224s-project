import sys

import numpy as np
from pandas_ml import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support

CLASS_LABELS = [
    'ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'
]

# load all predictions into a dictionary
predictions = {}
with np.load(sys.argv[1]) as data:
    for id, prediction in zip(data["speaker_ids"], data["predictions"]):
        predictions[int(id)] = CLASS_LABELS[prediction]

# import the correct labels
labels_data = np.genfromtxt("var/data/labels/dev/labels.dev.csv", dtype=None, names=True, delimiter=",")

# go through the labels and generate the correct arrays
speakers = [datum[0] for datum in labels_data]
y_true = [datum[3] for datum in labels_data]
y_pred = [predictions[i] for i in speakers]

# feed into sklearn
cm = ConfusionMatrix(y_true, y_pred)
print "=== CONFUSION MATRIX ==="
print cm
precisions, recalls, fscores, supports = precision_recall_fscore_support(y_true, y_pred)
print "=== METRICS ==="
for label, precision, recall, fscore, support in zip(CLASS_LABELS, precisions, recalls, fscores, supports):
    print "%s,%s,%s,%s,%s" % (label, precision, recall, fscore, support)
