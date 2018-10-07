USRA Summer 2018 - Multimodal Sarcasm Detection

Datasets from report:

- daria = dataset 1
- set1 = dataset 2
- set2 = dataset 3

File/folders included:

dariaAnnotations - annotations for dataset 1 (daria)
Data - folder with all the metadata, labels
doesnt_work - code that does not work
intensityContours - folder with intensity contours for each utterance
outputs - .arff files for all the experiments run
pitchContours - folder with pitch contours for each utterance
SemEval2018Task3_UCDCC - UCDCC outputs and SemEval gold labels
sentenceFeatures - output; sentence-level features for each utterance

compile.py - utility methods for compiling results into .csv and .arff files
test.py - main testing file; has separate methods for testing with datasets 1, 2, and 3
speech.py - collect speech features; implementation code
speech_utilities.py - contains classes and utility functions for speech.py

Notes:
- set1 and set2 don't have annotations, so the following don't work:
	- word-level pitch feature extraction
	- syllables/sec
	