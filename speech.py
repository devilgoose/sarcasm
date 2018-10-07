
import os

import speech_utilities as spf


# get sentence- and word-level sentence_features of the speech data
# requires 2 directories:
#   - directory with intensity contours, one file per sample
#   - directory with pitch contours, one file per sample

def get_speech_features(speech_data, annotations, dataset):

    sentence_features = {}
    current = os.getcwd()

    for d in speech_data:
        sentence_features[d[0]] = spf.SentenceFeatures(d, dataset)

    for line in annotations:
        name = line[0]
        del line[0]

        for word in line:
            if word == '[length]':
                sentence_features[name].words.append('#')
            elif word == '':
                pass
            else:
                sentence_features[name].words.append(word)

    # get pitch and intensity contours, calculate syllables per sec
    for (name, u) in sentence_features.items():
        os.chdir('pitchContours/' + dataset)
        u.parsePitch(name)

        os.chdir('../../intensityContours/' + dataset)
        u.parseIntensity(name)
        os.chdir(current)

        u.calc_syllables()

    print('normalizing based on speakers mean')

    # normalization of contours
    if dataset == 'set2':
        u.read_means()
    else:
        u.calcMeans()  # find means for each speaker

    for name, u in sentence_features.items():
        u.normalize()

    u.output_means()

    print('outputting features')

    os.chdir(current) # making sure working directory is as expected
    spf.write_sentence(sentence_features, dataset)

    #spf.to_binary(sentence_features.values(), dataset)

    data = []
    predict = []

    if dataset == 'daria':
        to_predict = 2
    elif dataset == 'set1':
        to_predict = 99
    else:
        to_predict = 0

    for name, u in sentence_features.items():
        utterance = u.features()
        if u.sarcasm != to_predict:
            utterance.append(name)
            data.append(utterance)
        else:
            predict.append(utterance)

    return list([data, predict])

def get_daria_features(speech_data, annotations):
    sentence_features = {}
    current = os.getcwd()

    for d in speech_data:
        sentence_features[d[0]] = spf.SentenceFeatures(d, 'daria')

    for line in annotations:
        name = line[0]
        del line[0]

        for word in line:
            if word == '[length]':
                sentence_features[name].words.append('#')
            elif word == '':
                pass
            else:
                sentence_features[name].words.append(word)

    # get pitch and intensity contours, calculate syllables per sec
    for (name, u) in sentence_features.items():
        os.chdir('pitchContours/Daria')
        u.parsePitch(name)

        os.chdir('../../intensityContours/daria')
        u.parseIntensity(name)
        os.chdir(current)

        u.calc_syllables()

    print('normalizing based on speakers mean')

    # normalization of contours

    u.calcMeans()  # find means for each speaker

    for name, u in sentence_features.items():
        u.normalize()

    print('splitting contours')

    for (name, u) in sentence_features.items():
        if (u.words == []):
            print(name)
        pitches = u.pitch
        intensities = u.intensity
        # parse pitch and intensity contours
        u.splitContours(pitches, 'p')
        u.splitContours(intensities, 'i')

    word_features = {}
    print('fitting to legendre polynomials')

    # get legendre polynomials
    for (name, u) in sentence_features.items():

        utterance = spf.WordFeatures(list([u.name, u.sarcasm, u.words]))

        errorp = errori = 0

        errorp = utterance.legendre(u.pitch, utterance.pcoefficients, 'p')
        errori = utterance.legendre(u.intensity, utterance.icoefficients, 'i')

        if errorp == -1 or errori == -1:
            print(name + ' is whack yo')

        utterance.updateTotals()

        word_features[name] = utterance

    word_features[name].ngrams(word_features)

    print('outputting features')

    os.chdir(current)  # making sure working directory is as expected

    # spf.to_binary(sentence_features.values(), dataset)

    data = []
    predict = []

    to_predict = 2

    for name, u in sentence_features.items():
        utterance = u.features()
        if u.sarcasm != to_predict:
            utterance.append(name)
            data.append(utterance)
        else:
            predict.append(utterance)

    return list([data, predict])