
import os
import keras

import speech as sp
import text as txt


# container class for all files in the program's subfolders
# each dictionary entry is a folder, and its value is a list of file names
class FileDict:
    def __init__(self):
        self.files = {}

    def sort(self):
        self.files['parsedAnnotations'].sort()
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


# classifies just speech
def test_speech():
    print('testing speech feature extraction - initializing...')

    current = os.getcwd()

    folders = []

    for (_, dirnames, _) in os.walk(current):
        folders = dirnames
        break

    fi = FileDict()

    for folder in folders:
        for (_, _, filenames) in os.walk(folder):
            fi.files[folder] = filenames

    print('getting data')

    # read metadata for list of word_features, create objects corresponding to each entry
    os.chdir('Metadata')
    corpus = []
    training_labels = []
    test_labels = []

    with open('data.csv', 'r') as f:
        lines = f.readlines()
        del lines[0]  # deleting header line
        for line in lines:
            data = [x.strip('\n') for x in line.split(',')]
            corpus.append(data)
            if 'daria' in data[0] :
                if data[-1] == 'Sarcastic':
                    training_labels.append(1)
                elif data[-1] == 'Sincere':
                    training_labels.append(0)
                else:
                    continue
            else:
                if data[-1] == 'Sarcastic':
                    test_labels.append(1)
                elif data[-1] == 'Sincere':
                    test_labels.append(0)
                else:
                    continue

    training_bin = keras.utils.to_categorical(training_labels)
    test_bin = keras.utils.to_categorical(test_labels)

    os.chdir(current)

    # word boundary annotations
    annotation_files = fi.parse('parsedAnnotations')
    annotations = []

    count = 0
    for lines in annotation_files:
        annotation = []
        (name, _) = fi.files['parsedAnnotations'][count].split('.')
        annotation.append(name)
        for line in lines:
            temp = [x.strip('\n') for x in line.split('\t')]
            annotation.append(temp)
        annotations.append(annotation)

        count += 1

    os.chdir(current)
    (train_samples, test_samples, predictions) = sp.get_speech_features(corpus, annotations)

    model = keras.Sequential()

    model.add(keras.layers.Dense(2, input_dim=12))
    model.add(keras.layers.Activation('softmax'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    model.fit(x=train_samples, y=training_bin, validation_split=0.33)

    performance = model.evaluate(x=test_samples,y=test_bin)

    results = model.predict(x=predictions)

    print(performance)

    return results



def test_text():
    print('testing text feature extraction - initializing...')

    current = os.getcwd()

    folders = []

    for (_, dirnames, _) in os.walk(current):
        folders = dirnames
        break

    fi = FileDict()

    for folder in folders:
        for (_, _, filenames) in os.walk(folder):
            fi.files[folder] = filenames

    print('getting data')

    # read metadata for list of word_features, create objects corresponding to each entry
    os.chdir('Metadata')
    corpus = []
    training_labels = []
    test_labels = []

    with open('data.csv', 'r') as f:
        lines = f.readlines()
        del lines[0]  # deleting header line
        for line in lines:
            data = [x.strip('\n') for x in line.split(',')]
            corpus.append(data)
            if 'daria' in data[0]:
                if data[-1] == 'Sarcastic':
                    training_labels.append(1)
                elif data[-1] == 'Sincere':
                    training_labels.append(0)
                else:
                    continue
            else:
                if data[-1] == 'Sarcastic':
                    test_labels.append(1)
                elif data[-1] == 'Sincere':
                    test_labels.append(0)
                else:
                    continue

    training_bin = keras.utils.to_categorical(training_labels)
    test_bin = keras.utils.to_categorical(test_labels)

    os.chdir(current)

    # word boundary annotations
    annotation_files = fi.parse('parsedAnnotations')
    annotations = []

    count = 0
    for lines in annotation_files:
        annotation = []
        (name, _) = fi.files['parsedAnnotations'][count].split('.')
        annotation.append(name)
        for line in lines:
            (_, _, temp) = [x.strip('\n') for x in line.split('\t')]
            annotation.append(temp)
        annotations.append(annotation)

        count += 1

    os.chdir(current)

    (model, training, test, predictions) = txt.get_text_features(annotations)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=[keras.metrics.categorical_accuracy])
    print(model.summary())
    print(model.layers, model.inputs, model.outputs)

    # fit to training data
    stats = model.fit([training[0], training[1]], [training_bin], epochs=6, validation_split=0.1)

    # evaluate on test data
    perf = model.evaluate([test[0], test[1]], [test_bin])
    print(perf)

    # output by making system predict outputs of test set
    predictions = model.predict([predictions[0], predictions[1]])


def test_fusion():
    pass

out = test_speech()

print(out)