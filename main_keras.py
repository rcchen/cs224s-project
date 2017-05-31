import os

import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from nltk import ngrams

# CLASS_LABELS = [
#     'ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'
# ]

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

BATCH_SIZE = 100
NUM_WORDS = 1000

def load_files_from_directory(path):
    data = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            contents = open("%s/%s" % (path, filename)).read()
            data.append(contents)
        else:
            continue
    return np.array(data)

def load_labels_from_file(path):
    labels = pd.read_csv(path)
    return [CLASS_LABELS[i] for i in np.concatenate(labels.as_matrix()[:, [3]]).ravel()]

print "Loading data..."
x_train = load_files_from_directory("var/data/essays/train/tokenized")
x_dev = load_files_from_directory("var/data/essays/dev/tokenized")
y_train = load_labels_from_file("var/data/labels/train/labels.train.csv")
y_dev = load_labels_from_file("var/data/labels/dev/labels.dev.csv")

NUM_CLASSES = len(np.unique(y_train))

print "Vectorizing sequence data..."
tokenizer = Tokenizer(num_words=NUM_WORDS, split=" ")
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.sequences_to_matrix(x_train, mode="freq")
x_dev = tokenizer.sequences_to_matrix(x_dev, mode="freq")

print "Convert class vector..."
y_train = to_categorical(y_train, NUM_CLASSES)
y_dev = to_categorical(y_dev, NUM_CLASSES)

print "Building model..."
model = Sequential()
model.add(Dense(units=64, input_shape=(NUM_WORDS,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=NUM_CLASSES))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_dev, y_dev))
