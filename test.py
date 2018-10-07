
import os
import string

import speech as sp

from compile import compile_weka
from compile import compile_csv


# container class for all files in the program's subfolders
# each dictionary entry is a folder, and its value is a list of file names
class FileDict:
    def __init__(self):
        self.files = {}
        self.current = os.getcwd()
        folders = []

        for (_, dirnames, _) in os.walk(self.current):
            folders = dirnames
            break

        for folder in folders:
            for (_, _, filenames) in os.walk(folder):
                self.files[folder] = filenames

        print('getting data')

    def sort(self):
        self.files['pitchContours'].sort()
        self.files['intensityContours'].sort()

        # reads lines from a directory stored in the dict, returns a list of lists

    # initial list of all of the files from the directory, each entry has a
    def parse(self, directory):
        os.chdir(directory)
        files = self.files[directory]
        output = []
        for fi in self.files[directory]:
            with open(fi, 'r') as f:
                entry = f.readlines()
            output.append(entry)
        os.chdir('..')
        return output


# testing speech on the daria corpus
def test_daria():
    print('testing daria corpus - initializing...')

    fi = FileDict()

    # read metadata for list of word_features, create objects corresponding to each entry
    os.chdir('Data')
    corpus = []
    annotations = []

    with open('DSC_data.csv', 'r') as f:
        lines = f.readlines()
        del lines[0]  # deleting header line
        index = 4
        label = -1
        for line in lines:
            data = [x.strip('\n') for x in line.split(',')]
            annotation = list(['daria' + str(index)])
            words = [x.strip('\n') for x in line.split('"')]

            words = [x.strip(string.punctuation).lower() for x in words[3].split()]
            annotation.extend(words)

            if data[-2] == 'Sarcastic':
                label = 1
            elif data[-2] == 'Sincere':
                label = 0
            else:
                label = 2

            corpus.append(list(['daria' + str(index), label]))
            annotations.append(annotation)

            if index != 78:
                index += 1
            else:
                index = 83

    os.chdir(fi.current)

    (samples, predictions) = sp.get_daria_features(speech_data=corpus, annotations=annotations)

    print('outputting to file...')
    os.chdir(fi.current + '/outputs/')

    compile_weka(samples, file_name='daria_word')

    return


def test_set1():
    print('testing speech features for dataset 1 - initializing...')

    fi = FileDict()

    # read metadata for list of word_features, create objects corresponding to each entry
    os.chdir('Data')
    corpus = []
    annotations = []

    with open('set1.csv', 'r') as f:
        lines = f.readlines()
        del lines[0]  # deleting header line

    for line in lines:

        annotation = []

        data = [x.strip('\n') for x in line.split(',')]
        corpus.append(list([data[0], data[-2]]))

        words = [x.strip(string.punctuation).lower() for x in data[1].split()]
        annotation.append(data[0])
        annotation.extend(words)

        annotations.append(annotation)

    # get the names of the speaker from each voice clip
    with open('names1.csv', 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        data = [x.strip('\n') for x in lines[i].split(',')]
        corpus[i].append(data[1])

    os.chdir(fi.current)

    (samples, predictions) = sp.get_speech_features(speech_data=corpus, dataset='set1', annotations=annotations)

    os.chdir(fi.current + '/outputs')

    compile_weka(file_name='set1_intent_binary', samples=samples)

    return



def test_set2():
    print('testing speech features for dataset 2 - initializing...')

    fi = FileDict()

    # read metadata for list of word_features, create objects corresponding to each entry
    os.chdir('Data')
    corpus = []
    annotations = []

    with open('set2.csv', 'r') as f:
        lines = f.readlines()
        del lines[0]  # deleting header line

    for line in lines:

        annotation = []

        data = [x.strip('\n') for x in line.split(',')]
        corpus.append(list([data[0], data[-1]]))

        words = [x.strip(string.punctuation).lower() for x in data[1].split()]
        annotation.append(data[0])
        annotation.extend(words)

        annotations.append(annotation)

    # get the names of the speaker from each voice clip
    with open('names2.csv', 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        data = [x.strip('\n') for x in lines[i].split(',')]
        corpus[i].append(data[1])

    os.chdir(fi.current)

    (samples, predictions) = sp.get_speech_features(speech_data=corpus, dataset='set2', annotations=annotations)

    os.chdir(fi.current + '/outputs')

    compile_csv(file_name='set2_turk', samples=samples)

    return


test_daria()
