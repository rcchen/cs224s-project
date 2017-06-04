import numpy as np

from keras import initializers
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from src.datasets.essays import load_essays_data, load_essays_pos
from src.datasets.labels import load_labels
from src.datasets.speech import load_speech_data, load_speech_pos

num_classes = 11
num_pos_tags = 45
num_tags = 876
hidden_dim = 25

def run():
    X_train, X_test = load_essays_pos()
    y_train, y_test = load_labels()

    # bin the POS tags
    # X_train = np.array([np.bincount(row, minlength=num_pos_tags) for row in X_train])
    # X_test = np.array([np.bincount(row, minlength=num_pos_tags) for row in X_test])

    X_train = np.array([np.pad(row, num_tags-len(row), 'constant')[num_tags-len(row):] if len(row) < num_tags else row[:num_tags] for row in X_train])
    X_test = np.array([np.pad(row, num_tags-len(row), 'constant')[num_tags-len(row):] if len(row) < num_tags else row[:num_tags] for row in X_test])

    print X_train.shape
    print X_test.shape

    X_train = X_train.reshape(X_train.shape[0], -1, 1)
    X_test = X_test.reshape(X_test.shape[0], -1, 1)

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')

    print X_train.shape
    print X_test.shape

    # one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Yer basic neural net (0.3755)
    # model.add(Dense, num_classes, input_dim=num_pos_tags)
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # create the model
    model = Sequential()
    model.add(Embedding(num_pos_tags, hidden_dim, input_length=num_tags))
    model.add(SimpleRNN(20,
                        input_shape=X_train.shape[1:]))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, nb_epoch=100, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # Classes
    classes = model.predict_classes(X_test)
    print classes

if __name__ == '__main__':
    run()
