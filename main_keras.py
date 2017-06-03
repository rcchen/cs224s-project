import os

import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from nltk import ngrams 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

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
    return np.concatenate(labels.as_matrix()[:, [3]]).ravel()

print "Loading data..."
x_train = load_files_from_directory("var/data/essays/train/tokenized")
x_dev = load_files_from_directory("var/data/essays/dev/tokenized")
y_train_raw = load_labels_from_file("var/data/labels/train/labels.train.csv")
y_dev_raw = load_labels_from_file("var/data/labels/dev/labels.dev.csv")

print "Vectorizing sequence data..."
tokenizer = Tokenizer(num_words=NUM_WORDS, split=" ")
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.sequences_to_matrix(x_train, mode="freq")
x_dev = tokenizer.sequences_to_matrix(x_dev, mode="freq")

print "Convert class vector..."
encoder = LabelEncoder()
encoder.fit(y_train_raw)

y_train_encoded = encoder.transform(y_train_raw)
y_dev_encoded = encoder.transform(y_dev_raw)

y_train_dummy = to_categorical(y_train_encoded)
y_dev_dummy = to_categorical(y_dev_encoded)

def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_shape=(1000,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def my_model():
    model = Sequential()
    model.add(Dense(units=64, input_dim=NUM_WORDS))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=11))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

print "Building model..."

model = my_model()

print x_train.shape
print y_train_dummy.shape

print np.count_nonzero(x_train)

print x_train[0]
print y_train_dummy[0]

history = model.fit(x_train, y_train_dummy,
                    batch_size=BATCH_SIZE,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_dev, y_dev_dummy))
