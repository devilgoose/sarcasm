import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import re

import numpy as np
import tensorflow as tf

# wrapper to turn tensorflow metric function into keras metric
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
	
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
	
# fragmenting and padding the sentences, return a large list of all the sentences
def splitSentences(toSplit, labels, output_labels):
	input1 = []
	input2 = []
	count = 0
	counts = [] # list of # fragments per sentence
	indexes = [] # index of the sentence
	
	# for each sentence
	for e in toSplit:
		combinations = (len(e) - 5) # find number of times to fragment sentence
		counts.append(combinations) 
		split = 3 # begin by splitting after 3rd word
		# if # tokens is less than 5, just put into both lists as is
		if combinations == 0:
			input1.append(e)
			input2.append(e)
			output_labels.append(labels[count])
			indexes.append(count)
		else:
			for i in range(combinations):	
				input1.append(e[:split])
				input2.append(e[split:])
				output_labels.append(labels[count])
				split += 1
				indexes.append(count)
		count += 1
	
	# pad all inputs
	pad1 = keras.preprocessing.sequence.pad_sequences(input1, maxlen=maxLength, padding='post', truncating='post')
	pad2 = keras.preprocessing.sequence.pad_sequences(input2, maxlen=maxLength, padding='post', truncating='post')
	return list([pad1, pad2, indexes, counts])
	
	
# ***** MAIN *******


# ********** PART 1: Data extraction & normalization

# initializing containers
sentences = []  # list of all sentences
sarcasm = [] # list of all labels
indexes = [] # list of indices

current = os.getcwd()

# extract all sentences
os.chdir('data')

print('extracting from corpus...')

data = []

# training set
with open('SemEval2018-T3-train-taskB.txt', 'r', encoding='utf8') as f:
	for line in f:
		n = [x.strip('\n') for x in line.split()]
		data.append(n)
		
	del data[0] # deleting the header
	while len(data) != 0:
		sentences.append(data[0][2:])
		sarcasm.append(int(data[0][1]))
		indexes.append(int(data[0][0]))
		del data[0]

# test set - could be read into separate lists?
with open('SemEval2018-T3_gold_test_taskB_emoji.txt', 'r', encoding='utf8') as f:
	for line in f:
		n = [x.strip('\n') for x in line.split('\t')]
		data.append(n)
			
	del data[0]
	while len(data) != 0:
		sentences.append(data[0][2:])
		sarcasm.append(int(data[0][1]))
		indexes.append(int(data[0][0]))
		del data[0]
			
os.chdir(current)

sarcasm = np.asarray(sarcasm) # turning sarcasm list into np array

tweetokenized = []
wordCount = 0
count = {} # keeping count of number of occurrences for each word
                

for s in sentences:
	temp = []
	
	for w in s:
		if '#' in w: # hashtags
			w = w.replace('#','')
			w = [a.lower() for a in re.split('([A-Z][a-z]+)', w) if a]
			wordCount += len(w)
			temp.extend(w)
		elif '@' in w: # usernames
			wordCount += 1
			temp.append('[user]')
		elif '://' in w: # URLs
			temp.append('[URL]')
		elif "'" in w:
			if (w.endswith("n't")): # the n't case - don't, can't, etc.
				word1 = w[:-3]
				word2 = "n't"
			else:
				(word1, word2) = w.split("'", 1)
				if (word2 != ''): # checking for plural possessive, eg. the States'
					word2 = "'" + word2
			temp.append(word1.lower())
			temp.append(word2.lower())
			wordCount += 2
		else:
			temp.append(w.lower())
			wordCount += 1
	for w in temp:
		if w in count:
			count[w] += 1
		else:
			count[w] = 1
	tweetokenized.append(temp)

# get length of sentences to input - is the mean wordcount, based on UCDCC SemEval 2018 work
maxLength = int(wordCount/len(sentences)) 
print(maxLength)

# tokenization using keras - replaces the tokens w\ their index in the master list of tokens
t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(tweetokenized)
encoded = t.texts_to_sequences(tweetokenized)

vocabSize = len(t.word_index) + 1 # size of the vocab list

# remove once-occurring words, also usernames and URLs
# might want to keep username
words = {v: k for k, v in t.word_index.items()} # reversing word index dict
for i in encoded:
	for w in i:
		if count[words[w]] <= 1 or w == '[user]' or w == '[URL]':
			w = 0


# get word embeddings - 50-dim from Twitter
print('loading word embeddings...')

embeddings = {}
vectors = []
with open('glove.27b.txt', 'r', encoding='utf8') as f:
	for line in f:
		n = [x.strip('\n') for x in line.split()]
		n = [float(x) if x != n[0] else x for x in n]
		if n[0] in t.word_index:
			vectors = n[1:]
			embeddings[n[0]] = vectors

#making final matrix to pass into embedding layer			
embeddingMat = np.zeros((vocabSize, 50))

for word, i in t.word_index.items():
	if word in embeddings:
		vector = embeddings[word]
		if vector is not None:
			embeddingMat[i] = vector
	
print('done embedding')


# ***** PART 2: MODEL - semEval UCDCC model
# splitting the inputs into different combinations
print('splitting the inputs...')

sarcasm_training = [] # list of sarcasm labels for training set
sarcasm_test = [] # sarcasm labels for test set

# fragment the sentences
(training1, training2, indexes_train, _) = splitSentences(encoded[:3834], sarcasm[:3834], sarcasm_training)
(test1, test2, indexes_test, counts_test) = splitSentences(encoded[3834:], sarcasm[3834:], sarcasm_test)

# convert sarcasm labels to one-hot encoded vectors
training_bin = keras.utils.to_categorical(sarcasm_training)
test_bin = keras.utils.to_categorical(sarcasm_test)

# layers
inputLayer1 = keras.layers.Input(shape=(maxLength,), batch_shape=(None,12), name='input1') # input 1
inputLayer2 = keras.layers.Input(shape=(maxLength,), batch_shape=(None,12), name='input2') # input 2
# embedding layers - might merge into one later
embed1 = keras.layers.Embedding(vocabSize, 50, 
								weights=[embeddingMat], 
								input_length=maxLength, 
								trainable=False)(inputLayer1)
embed2 = keras.layers.Embedding(vocabSize, 50, 
								weights=[embeddingMat], 
								input_length=maxLength, 
								trainable=False)(inputLayer2)
# LSTM layers
LSTM1 = keras.layers.LSTM(units=32, activation='sigmoid', kernel_initializer='glorot_normal', recurrent_dropout=0.5)(embed1)
LSTM2 = keras.layers.LSTM(units=32, activation='sigmoid', kernel_initializer='glorot_normal', recurrent_dropout=0.5)(embed2)

# subtract output of 2 LSTM layers
sub = keras.layers.Subtract()([LSTM1, LSTM2])

# classification - dense NN
dnn = keras.layers.Dense(units=16, kernel_initializer='glorot_normal', activation='relu')(sub)
dnn = keras.layers.Dense(4, kernel_initializer='glorot_normal', activation='relu')(dnn)

# output - softmax
soft = keras.layers.Softmax()(dnn)

# the stupid goddang metrics that don't work
precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

# initialize and compile model
model = keras.models.Model(inputs=[inputLayer1, inputLayer2], outputs=[soft])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[keras.metrics.categorical_accuracy])
print(model.summary())

# fit to training data
stats = model.fit([training1, training2], [training_bin], validation_split=0.1)

# evaluate on test data
perf = model.evaluate([test1, test2], [test_bin])

# output by making system predict outputs of test set
predictions = model.predict([test1, test2])

# checking how many items in test set
print(perf)

# output predictions to csv file
with open('predictions.csv', 'w') as f:
	for i in range(len(predictions)):
		f.write(str(sarcasm_test[i]) + ',' + str(indexes_test[i]))
		for x in predictions[i]:
			f.write(',' + str(x))
		f.write('\n')



