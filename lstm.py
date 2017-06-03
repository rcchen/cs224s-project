import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from src.datasets.essays import load_essays_data
from src.datasets.labels import load_labels
from src.datasets.speech import load_speech_data

num_classes = 11
top_words = 1000

def run():

    X_train, X_test = load_speech_data(num_words=top_words)
    y_train, y_test = load_labels()

    # truncate and pad input sequences
    maxlen = 500
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    # one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # create the model
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=maxlen))
    model.add(LSTM(20))
    model.add(Dense(11, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, nb_epoch=1, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # Classes
    classes = model.predict_classes(X_test)
    print classes

if __name__ == '__main__':
    run()
