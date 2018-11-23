USRA Summer 2018 - Multimodal Sarcasm Detection

no metadata so technically doesn't run but the complete outputs are in the outputs file
also all extracted contours are included

File/folders included:

parsedAnnotations - annotations for dataset 1 (daria)
doesnt_work - code that did not work
intensityContours - folder with intensity contours for each utterance
outputs - .arff files for all the experiments run
pitchContours - folder with pitch contours for each utterance
sentenceFeatures - output; sentence-level features for each utterance

compile.py - utility methods for compiling results into .csv and .arff files
test.py - main testing file; has separate methods for testing with datasets 1, 2, and 3
speech.py - collect speech features; implementation code
speech_utilities.py - contains classes and utility functions for speech.py

Notes:
- set1 and set2 don't have annotations, so the following don't work:
	- word-level pitch feature extraction
	- syllables/sec
	