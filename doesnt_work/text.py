import keras
import os
import re
import numpy as np
import tensorflow as tf

import text_utilities as util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ********** PART 1: Data extraction & normalization


# due to scale of features, doesn't return full features
# using LSTM, scales feature dimensionality down
def get_text_features(text_data, train_labels, test_labels):
    # initializing containers

    current = os.getcwd()

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    (treated, wordCount) = util.treat_tweets(text_data) # treating the tweets

    # get length of text_data to input - is the mean wordcount, based on UCDCC SemEval 2018 work
    max_length = int(wordCount / len(text_data))

    print(max_length)
    print(text_data[-1])
    print(text_data[12])
    print(treated[-1])
    print(treated[12])

    (encoded, embedding_mat, vocabSize) = util.embed(treated)

    # ***** PART 2: MODEL - semEval UCDCC model
    # splitting the inputs into different combinations
    print('splitting the inputs...')

    training_set = []
    training_labels = []
    test_set = []
    prediction_set = []

    # fragment the text_data
    (training1, training2, indexes_train, labels_training) = util.splitSentences(training_set, training_labels,  max_length)
    (test1, test2, indexes_test, labels_test) = util.splitSentences(test_set, test_labels, max_length)
    (pred1, pred2, indexes_pred, _) = util.splitSentences(prediction_set, prediction_labels, max_length)

    # balance the dataset
    (training1, training2, indexes_train, labels_training) = util.balance(training1, training2, indexes_train, labels_training, 4)
    (test1, test2, indexes_test, labels_test) = util.balance(test1, test2, indexes_test, labels_test, 4)

    # convert sarcasm labels to one-hot encoded vectors
    training_bin = keras.utils.to_categorical(labels_training)
    test_bin = keras.utils.to_categorical(labels_test)

    print('building model...')

    # layers
    inputLayer1 = keras.layers.Input(shape=(max_length,))  # input 1
    inputLayer2 = keras.layers.Input(shape=(max_length,))  # input 2
    # embedding layers - might merge into one later
    embed1 = keras.layers.Embedding(vocabSize, 25,
                                    weights=[embedding_mat],
                                    input_length=max_length,
                                    trainable=False, mask_zero=True)(inputLayer1)
    embed2 = keras.layers.Embedding(vocabSize, 25,
                                    weights=[embedding_mat],
                                    input_length=max_length,
                                    trainable=False, mask_zero=True)(inputLayer2)
    # LSTM layers
    LSTM1 = keras.layers.LSTM(units=32, activation='sigmoid',
                              kernel_initializer='glorot_normal',
                              recurrent_dropout=0.5)(embed1)
    LSTM2 = keras.layers.LSTM(units=32, activation='sigmoid',
                              kernel_initializer='glorot_normal',
                              recurrent_dropout=0.5)(embed2)

    # subtract output of 2 LSTM layers
    sub = keras.layers.Subtract()([LSTM1, LSTM2])
    # classification - dense NN
    # dnn = keras.layers.Dense(units=4, kernel_initializer='glorot_normal', activation='relu')(sub)

    soft = keras.layers.Dense(units=4, kernel_initializer='glorot_normal', activation='softmax')(sub)

    adam = keras.optimizers.Adam(lr=0.001)

    # initialize and compile model
    model = keras.models.Model(inputs=[inputLayer1, inputLayer2], outputs=[soft])

    return list([model, list([training1, training2]), list([test1, test2]), list([pred1, pred2])])
