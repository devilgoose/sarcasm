
import os

import keras
import tensorflow as tf
import nltk

import re
import numpy as np
import math

from SpellingReplacer import spell_correction

from feel_the_positivity import pos_neg_replacer

from nltk.corpus import wordnet as wn

import textblob

nltk.download('wordnet')

def treat_tweets(sentences):
    sentences = parse_emojis(sentences)
    (formatted, wordCount) = formatting(sentences)
    return list([formatted, wordCount])


# parse emojis, replacing emoji in sentence with the 'official' name
def parse_emojis(sentences):
    emojis = {}

    print('reading emojis...')
    # reading emojis from text file
    with open('data/emojis.txt', 'r', encoding='utf8') as f:
        for line in f:
            (emoji, name) = [x.strip('\n') for x in line.split(',', 1)]
            emojis[emoji] = '_'.join(name.split(' '))

    for sentence in sentences:
        count = 0
        for w in sentence: # for each word
            newstring = w[0].strip(':')
            for char in w[0]:
                if char in emojis:
                    newstring = newstring.replace(char, emojis[char] + ' ')
            sentence[count][0] = newstring
            count += 1

    return sentences


# replaces tweet-specific formatting, eg. usernames, hashtags, URLs, and normalizes spelling
# returns the transformed sentences and the total wordCount
def formatting(sentences):

    tweetokenized = []
    wordCount = 0
    count = {}  # keeping count of number of occurrences for each word

    for s in sentences:
        temp = []

        for w in s:
            if "'" in w: # contractions - need to split for GloVe
                if w.endswith("n't"):  # the n't case - don't, can't, etc.
                    word1 = w[:-3]
                    word2 = "n't"
                else:
                    (word1, word2) = w.split("'", 1)
                    if word2 != '':  # checking for plural possessive, eg. the States'
                        word2 = "'" + word2
                temp.append(word1.lower())
                temp.append(word2.lower())
                wordCount += 2
            else:
                word = w[0].replace(':', '')
                temp.append(word.lower())
                wordCount += 1
        temp = [x for x in temp if x]
        tweetokenized.append(temp)

    spelling = []
    for s in tweetokenized:
        spelling.append(spell_correction(' '.join(s)))
    sentences = []
    for s in spelling:
        sentences.append(pos_neg_replacer(s))
    tweetokenized = remove_one_off(sentences)

    return list([tweetokenized, wordCount])


# replace words with their wordnet "categories" - first hypernym based on first synset of word's pos tag
# after replacement, calls treat_tweets to parse sentences for further processing
def categorize(sentences):

    posentences = []
    for s in sentences:
        temp_sentence = []
        temp_word = []
        for w in s:
            if (w[1] == 'N' or w[1] == '^') and w[2] > 0.9:
                temp_word = list([w[0], wn.NOUN])
            elif w[1] == 'V' and w[2] > 0.9:
                temp_word = list([w[0], wn.VERB])
            elif w[1] == 'A' and w[2] > 0.9:
                temp_word = list([w[0], wn.ADJ])
            elif w[1] == 'R' and w[2] > 0.9:
                temp_word = list([w[0], wn.ADV])
            else:
                temp_word = list([w[0]])
            temp_sentence.append(temp_word)

        posentences.append(temp_sentence)

    new_sentences = []

    i = j = 0
    for s in posentences:
        temp_sentence = []
        j = 0
        if len(s) != len(posentences[i]):
            print('so this isnt going to work', i)
        for w in s:
            if w[0] != '':
                if len(w) > 1:
                    pos = w[1]
                    uh = wn.synsets(w[0][0], pos=pos)
                else:
                    uh = []

                if uh != []:
                    try:
                        hypernym = list(uh)[0].lemma()
                        print(hypernym)
                        hyp_name = list(hypernym)[0].name().split('.')[0].replace('_', ' ')
                        temp_sentence.append(hyp_name)
                    except:
                        temp_sentence.append(w[0])
                else:
                    temp_sentence.append(w[0])

                j += 1


        temp_sentence = ' '.join(temp_sentence)
        temp_sentence = temp_sentence.split(' ')

        new_sentences.append(temp_sentence)
        i += 1
    return new_sentences


def remove_one_off(sentences):
    count = {}
    for s in sentences:
        for w in s.split(' '):
            if w in count:
                count[w] += 1
            else:
                count[w] = 1

    i = j = 0
    for s in sentences:
        j = 0
        new_sentence = []
        for w in s.split(' '):
            if count[w] > 2:
                new_sentence.append(w)
            else:
                continue
            j += 1
        i += 1
    return sentences

# fragmenting and padding the sentences, returns two sets of fragments + labels and indexes corresponding to the
# newly extended dataset
def splitSentences(toSplit, labels, maxLength):
    input1 = []
    input2 = []
    count = 0
    indexes = []  # index of the sentence

    output_labels = []

    # for each sentence
    for e in toSplit:
        for w in e:
            if math.isnan(w) is True:
                print('heck')
        combinations = (len(e) - 5)  # find number of times to fragment sentence
        split = 3  # begin by splitting after 3rd word
        # if # tokens is less than 5, just put into both lists as is
        if combinations == 0:
            input1.append(e)
            input2.append(e)
            indexes.append(count)
            output_labels.append(labels[count])
        else:
            for i in range(combinations):
                input1.append(e[:split])
                input2.append(e[split:])
                split += 1
                indexes.append(count)
                output_labels.append(labels[count])
        count += 1

    # pad all inputs
    pad1 = keras.preprocessing.sequence.pad_sequences(input1, maxlen=maxLength, padding='post', truncating='post')
    pad2 = keras.preprocessing.sequence.pad_sequences(input2, maxlen=maxLength, padding='post', truncating='post')
    return list([pad1, pad2, indexes, output_labels])


def embed(treated):

    # tokenization using keras - replaces the tokens w\ their index in the master list of tokens
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(treated)
    encoded = t.texts_to_sequences(treated)
    print(encoded[-1])

    vocabSize = len(t.word_index) + 1  # size of the vocab list

    # remove once-occurring words, also usernames and URLs
    # might want to keep username

    words = {v: k for k, v in t.word_index.items()}  # reversing word index dict

    j = k = 0
    for i in encoded:
        for w in i:
            if w == '[user]' or w == '[URL]':
                w = 0

    # get word embeddings - 50-dim from Twitter
    print('loading word embeddings...')

    embeddings = {}
    vectors = []

    with open('glove.25d.txt', 'r', encoding='utf8') as f:
        for line in f:
            n = [x.strip('\n') for x in line.split()]
            n = [float(x) if x != n[0] else x for x in n]
            if n[0] in t.word_index:
                vectors = n[1:]
                embeddings[n[0]] = vectors

    # making final matrix to pass into embedding layer
    embedding_mat = np.zeros((vocabSize, 25))

    for word, i in t.word_index.items():
        if word in embeddings:
            vector = embeddings[word]
            if vector is not None:
                embedding_mat[i] = vector
            else:
                print(word)

    print('done embedding')

    return list([encoded, embedding_mat, vocabSize])


# up-sample dataset to balance out categories
def balance(samples1, samples2, indexes, labels, numcats):
    counts = np.zeros(numcats)
    for s in range(len(labels)):
        counts[labels[s]] += 1
    print(counts)

    #up-sample
    maxSamples = max(counts)
    tosample = np.zeros(numcats)
    num_duplicates = np.zeros(numcats, dtype=int)

    for i in range(numcats):
        tosample[i] = maxSamples - counts[i]
        num_duplicates[i] = int(tosample[i]/counts[i]) + 1

    newSamples1 = list(samples1)
    newSamples2 = list(samples2)
    newLabels = labels
    newIndexes = indexes

    for s in range(len(samples1)):
        if tosample[labels[s]] != 0:
            for i in range(num_duplicates[labels[s]]):
                newSamples1.append(samples1[s])
                newSamples2.append(samples2[s])

                newLabels.append(labels[s])
                newIndexes.append(indexes[s])

                tosample[labels[s]] -= 1

    newSamples1 = np.asarray(newSamples1)
    newSamples2 = np.asarray(newSamples2)

    counts = np.zeros(numcats)
    for s in range(len(labels)):
        counts[labels[s]] += 1
    print(counts)

    return list([newSamples1, newSamples2, newIndexes, newLabels])


pass
