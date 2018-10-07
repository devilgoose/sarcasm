
# some utility functions and classes for extracting the speech features

import math
import numpy as np
import os

import scipy.cluster as clust
import scipy.spatial.distance as dst

from nltk.corpus import cmudict

d = cmudict.dict()


# container class for contours, including sentence-level statistical features
class SentenceFeatures:
    speakers = {}  # dict to store mean pitch and intensity per file, one entry for each file
    means = {}  # dict to store mean pitch for speaker as a whole

    def __init__(self, data, dataset):
        self.length = 0.0
        self.words = []
        self.pitch = []
        self.intensity = []

        if dataset == 'daria':
            s = data[0]
            self.name = ''.join([i for i in s if not i.isdigit()])  # speaker name
        else:
            self.name = data[2]

        # initialize speakers dict
        if self.name in self.speakers.keys():
            pass
        else:
            self.speakers[self.name] = {'pitch': [], 'intensity': [], 'sarcasm': []}

        self.sarcasm = int(data[1])

        # sentence-level features
        self.meanPitch = 0.0
        self.pitchSD = 0.0
        self.pRange = 0.0
        self.meanInten = 0.0
        self.iRange = 0.0
        self.syllables = 0.0

    def parsePitch(self, name):
        pitchesNoSilences = []
        sumPitch = 0.0
        count = 0

        with open(name + '.f0', 'r') as f:
            for line in f:
                (pitch, prob, _, _) = line.split()
                pitch = float(pitch)
                if pitch != 0:
                    pitch = math.log10(float(pitch))

                self.pitch.append(pitch)  # complete list of values to be normalized later

                if (prob != 0):
                    pitchesNoSilences.append(pitch)  # add value to list for range and SD
                    sumPitch += pitch
                    count += 1

        # mean pitch
        self.meanPitch = sumPitch / count
        # add mean to overall list
        self.speakers[self.name]['pitch'].append(self.meanPitch)
        self.speakers[self.name]['sarcasm'].append(self.sarcasm)

        # pitch SD
        pitchVarSum = 0.0

        for p in pitchesNoSilences:
            pitchVarSum += pow(p - (self.meanPitch), 2)

        self.pitchSD = math.sqrt(pitchVarSum / len(pitchesNoSilences))

        # pitch range
        pitchesNoSilences.sort()

        # remove top and bottom 5.5% following Rachel's work - meant to remove outliers
        cutoff = int(len(pitchesNoSilences) * 0.055)
        del pitchesNoSilences[0:cutoff]
        del pitchesNoSilences[-cutoff:]

        # since already sorted, just get the first and last element of the list for min and max
        self.pRange = (pitchesNoSilences[-1] - pitchesNoSilences[0])

    def parseIntensity(self, name):

        intensities = []
        sumInten = 0.0
        count = 0

        with open(name + '.txt', 'r') as f:
            for line in f:
                self.intensity.append(float(line))
                intensities.append(float(line))

                sumInten += float(line)
                count += 1

        # mean intensity
        self.meanInten = sumInten / count

        # intensity range
        intensities.sort()

        # remove top and bottom 5.5% following Rachel's work - meant to remove outliers
        cutoff = int(len(intensities) * 0.055)
        del intensities[0:cutoff]
        del intensities[-cutoff:]

        # since already sorted, just get the first and last element of the list for min and max
        self.iRange = (intensities[-1] - intensities[0])

        self.speakers[self.name]['intensity'].append(self.meanInten)

    def calc_syllables(self):
        syllables = 0.0
        word_count = 0
        for word in self.words:
            if word[0] != '#':
                if nsyl(word[0]) != -1:
                    syllables = syllables + nsyl(word[0])
                word_count += 1

        self.syllables = syllables/word_count

    # calculate means
    def calcMeans(self):
        for (speaker, d) in self.speakers.items():
            sumPitch = 0
            sumInten = 0
            length = len(d['pitch'])
            count = 0
            for x in range(0, length):
                if d['sarcasm'][x] == 2 or d['sarcasm'][x] == 4:
                    sumPitch += d['pitch'][x]
                    sumInten += d['intensity'][x]
                    count += 1
            self.means[speaker] = list([sumPitch / count, sumInten / count])

    def output_means(self):
        with open('speaker_means.txt', 'w') as f:
            for (speaker, means) in self.means.items():
                f.write(str(speaker) + '\t' + str(means[0]) + '\t' + str(means[1]) + '\n')

    def read_means(self):
        with open('Data/speaker_means.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                (name, pitch, intensity) = [x.strip('\n') for x in line.split()]
                self.means[name] = list([float(pitch), float(intensity)])

        for (speaker, d) in self.speakers.items():
            if speaker not in self.means:
                sumPitch = 0
                sumInten = 0
                length = len(d['pitch'])
                count = 0
                for x in range(0, length):
                    if d['sarcasm'][x] == 2 or d['sarcasm'][x] == 4:
                        sumPitch += d['pitch'][x]
                        sumInten += d['intensity'][x]
                        count += 1
                self.means[speaker] = list([sumPitch / count, sumInten / count])

    # normalize the pitch and intensity contour values stored in the object
    # based on means calculated in calcMeans
    def normalize(self):
        norm_pitch = []
        for x in self.pitch:
            if x != 0:
                norm_pitch.append((x - self.means[self.name][0])/self.means[self.name][0])
            else:
                norm_pitch.append(0)

        self.pitch = norm_pitch

        norm_inten = []
        for x in self.intensity:
            if x >= 0:
                norm_inten.append((x - self.means[self.name][1]) / self.means[self.name][1])
            else:
                norm_inten.append(0)

        self.intensity = norm_inten

        self.meanPitch = (self.meanPitch - self.means[self.name][0]) / self.means[self.name][0]
        self.meanInten = (self.meanInten - self.means[self.name][1]) / self.means[self.name][1]
        self.pitchSD = (self.pitchSD - self.means[self.name][0]) / self.means[self.name][0]
        self.pRange = (self.pRange - self.means[self.name][0]) / self.means[self.name][0]
        self.iRange = (self.iRange - self.means[self.name][1]) / self.means[self.name][1]

        # divide the contours based on timings, store mapped to a time interval of [-1, +1]
        # if zeroes exist, are also stored, but will be removed later
        # also gets the mean of the contour,
    def splitContours(self, contour, ty):
        # get a measure of what samples correspond to what times
        if (self.length == 0):
            return
        if (ty == 'p'):
            self.pitch = []
        elif (ty == 'i'):
            self.intensity = []
        samplesPerSec = float(len(contour) + 1) / self.length

        start = 0

        for wrd in self.words:
            w = wrd[0]
            l = (wrd[2] - wrd[1])
            # get start and end times of sample,
            # start and end indices within contour list,
            # interval length
            samples = int(samplesPerSec * l)
            interval = (2 / (samplesPerSec * l))
            end = start + samples
            c = []

            # checking if the word should be ommitted - if so, ignore the contours for that interval
            if w == '#':
                pass
            else:

                tempCont = contour[start:end]

                count = 0
                for i in tempCont:
                    c.append(list([(-1 + interval * count), i]))
                    count += 1

            if ty == 'p':
                self.pitch.append(c)
            elif ty == 'i':
                self.intensity.append(c)
            start = end

    def features(self):
        return list([self.meanPitch, self.pRange, self.pitchSD, self.meanInten, self.iRange, self.syllables, self.sarcasm])


# container class for the legendre polynomial fit coefficients, as well as those resulting features
class WordFeatures:
    # stores data over entire corpus
    # total legendre polynomial coefficients for clustering
    totPCoefficients = []
    totICoefficients = []

    def __init__(self, data):
        self.words = data[2]
        self.pcoefficients = []  # pitch contour legendre polynomial coefficients; list of arrays, one entry per word
        self.icoefficients = []  # intensity contour legendre polynomial coefficients; list of arrays, one entry per word

        self.plabels = []  # list of pitch labels for each word, either '0', '1' or '2'
        self.ilabels = []  # list of intensity labels for each word, either '0', '1' or '2'

        self.puni = []  # frequency of each pitch label in utterance, measured by percentage
        self.iuni = []  # frequency of each intensity label in utterance, measured by percentage

        self.pbi = []  # bigram pitch probabilities for the utterance, depends on whether sarcastic or sincere
        self.ibi = []  # bigram intensity probabilities for utterance, same as pbi

        self.perplexities = []  # perplexity of the utterance - first entry is pitch, then intensity

        self.missing = False  # if there are missing contour values in the utterance, then raise this flag to exclude from bigram calculations

        self.name = data[0]  # speaker name

        # sarcasm label - ambiguous files are marked to be thrown away later
        self.sarcasm = data[1]

    def updateTotals(self):
        self.totICoefficients.extend(self.icoefficients)
        self.totPCoefficients.extend(self.pcoefficients)


    # from contour values, get legendre polynomial fit
    @staticmethod
    def legendre(contour, output, ty):
        num = 0
        for c in contour:
            xes = []
            ys = []
            if (len(c) != 0):
                for x in c:
                    if x[1] == 0.0:
                        c.remove(x)
                    else:
                        xes.append(x[0])
                        ys.append(x[1])

                coefficients = []

                if (len(xes) != 0):
                    # fit contour to legendre polynomial
                    coefficients = np.polynomial.legendre.legfit(xes, ys, 2)

                output.append(coefficients)
                for co in coefficients:
                    if abs(co) > 30:
                        num = -1
        return num

    @staticmethod
    def ngrams(utterances):

        p_clusters = Centroids(utterances['daria10'].totPCoefficients)
        i_clusters = Centroids(utterances['daria10'].totICoefficients)

        # output for debugging
        with open('centroids.txt', 'w') as f:
            f.writelines(str(p_clusters.A) + '\n')
            f.writelines(str(p_clusters.B) + '\n')
            f.writelines(str(p_clusters.C) + '\n\n')
            f.writelines(str(i_clusters.A) + '\n')
            f.writelines(str(i_clusters.B) + '\n')
            f.writelines(str(i_clusters.C) + '\n\n')

        # initialize counters
        # bigram
        # rows are preceding word, cols are current word
        pbigram_sa = np.zeros([3, 3])
        ibigram_sa = np.zeros([3, 3])
        pbigram_si = np.zeros([3, 3])
        ibigram_si = np.zeros([3, 3])
        pbigram_amb = np.zeros([3, 3])
        ibigram_amb = np.zeros([3, 3])

        for (name, u) in utterances.items():

            # use euclidean distance algorithm from numpy, label words with closest centroid
            count = 0
            for wrd in u.words:
                if (wrd[0] != '#'):
                    plabel = p_clusters.eucDist(u.pcoefficients[count])
                    if (plabel == -1):
                        u.missing = True
                        break
                    else:
                        u.plabels.append(plabel)
                    ilabel = i_clusters.eucDist(u.icoefficients[count])
                    if ilabel == -1:
                        u.missing = True
                        break
                    else:
                        u.ilabels.append(ilabel)
                    count += 1

            if u.missing is not True:
                # create unigram models for each sentence based on labels

                u.puni = p_clusters.unigram(u.plabels, 'p')
                u.iuni = i_clusters.unigram(u.ilabels, 'i')

                count = 0

                curr_plabel = u.plabels[0]
                prev_plabel = -1

                curr_ilabel = u.ilabels[0]
                prev_ilabel = -1

                # update bigram counts
                while (count < (len(u.plabels) - 1)):
                    if (count != 0):
                        if (u.sarcasm == 1):
                            pbigram_sa[curr_plabel][prev_plabel] += 1
                            ibigram_sa[curr_ilabel][prev_ilabel] += 1
                        elif (u.sarcasm == 0):
                            pbigram_si[curr_plabel][prev_plabel] += 1
                            ibigram_si[curr_ilabel][prev_ilabel] += 1
                        elif (u.sarcasm == 2):
                            pbigram_amb[curr_plabel][prev_plabel] += 1
                            ibigram_amb[curr_plabel][prev_plabel] += 1
                        else:
                            print("error in data processing!!! pls fix")

                    count += 1

                    prev_plabel = curr_plabel
                    curr_plabel = u.plabels[count]

                    prev_ilabel = curr_ilabel
                    curr_ilabel = u.ilabels[count]
            else:
                u.puni = list([0.0, 0.0, 0.0])
                u.iuni = list([0.0, 0.0, 0.0])

        unigramPTotals = list([p_clusters.totalA, p_clusters.totalB, p_clusters.totalC])
        unigramITotals = list([i_clusters.totalA, i_clusters.totalB, i_clusters.totalC])

        # normalizing bigrams with unigrams

        normalize(pbigram_sa, unigramPTotals)
        normalize(pbigram_si, unigramPTotals)
        normalize(ibigram_sa, unigramITotals)
        normalize(ibigram_si, unigramITotals)
        normalize(pbigram_amb, unigramPTotals)
        normalize(ibigram_amb, unigramITotals)

    def features(self):
        to_return = []
        to_return.extend(self.puni)
        to_return.extend(self.iuni)
        to_return.extend(self.perplexities)
        return to_return


# helper class for storing centroids and centroid counts of coefficients in word_features class
# one object used for pitch, one for intensity
class Centroids:

    def __init__(self, data):
        # initialize all centroids, as well as distortion storage place
        self.A = []
        self.B = []
        self.C = []
        self.distortion = 0.0

        self.totalA = 0
        self.totalB = 0
        self.totalC = 0

        self.kmeansAlg(data)

    def kmeansAlg(self, l):
        # convert list to array
        l[:] = [x for x in l if x != []]

        features = np.asarray(l)
        # pass through clustering algorithm
        clusters, self.distortion = clust.vq.kmeans(features, 3)

        # put clusters into A, B, and C
        self.A = clusters[0, :]
        self.B = clusters[1, :]
        self.C = clusters[2, :]

    def eucDist(self, point):
        if (point == []):
            return -1
        adist = dst.euclidean(self.A, point)
        bdist = dst.euclidean(self.B, point)
        cdist = dst.euclidean(self.C, point)

        m = min(adist, bdist, cdist)

        if (m == adist):
            return 0
        elif m == bdist:
            return 1
        elif m == cdist:
            return 2

    # unigram is percentage of each centroid represented in the graph
    def unigram(self, labels, type):
        acount = bcount = ccount = 0
        if len(labels) == 0:
            return list([0.0, 0.0, 0.0])

        for l in labels:
            if (l == 0):
                acount = acount + 1
            elif (l == 1):
                bcount = bcount + 1
            elif (l == 2):
                ccount = ccount + 1
            else:
                pass

        self.totalA += acount
        self.totalB += bcount
        self.totalC += ccount

        length = len(labels)
        return list([acount / length, bcount / length, ccount / length])


# gets number of syllables per word - have to enter in exceptions manually
def nsyl(word):

    try:
        sum = 0
        for x in d[word.lower()]:
            sum = sum + len(list(y for y in x if y[-1].isdigit()))
        return sum
    except KeyError:
        # if word not found in cmudict
        if (word == "Monic"):
            return 2
        elif (word == 'favourite'):
            return 3
        elif (word == 'HMV'):
            return 3
        elif (word == 'intolerabler'):
            return 5
        elif (word == 'trodsky'):
            return 2
        elif (word == 'swingin'):
            return 2
        elif (word == 'weirded'):
            return 2
        elif (word == 'hottie'):
            return 2
        elif (word == 'pinups'):
            return 2
        elif (word == 'dreamcoat'):
            return 2
        elif (word == "saleman's"):
            return 2
        elif (word == 'woodshop'):
            return 2
        elif (word == 'thats'):
            return 1
        elif (word == 'retartant'):
            return 3
        elif (word == 'burgular'):
            return 3
        elif (word == 'FDR'):
            return 3
        elif (word == 'hairstyles'):
            return 2
        elif (word == 'sublimate'):
            return 3
        elif (word == 'astrophysicists'):
            return 5
        elif (word == 'streetcop'):
            return 2
        elif (word == 'taudry'):
            return 2
        elif (word == 'furballs'):
            return 2
        elif (word == 'apperance'):
            return 3
        elif (word == 'RTC'):
            return 3
        elif (word == 'Blige'):
            return 1
        elif (word == 'enchiladas'):
            return 4
        elif (word == "Weston's"):
            return 2
        elif (word == 'singularity'):
            return 5
        elif (word == 'soundcloud'):
            return 2
        elif (word == 'sorta'):
            return 2
        elif (word == 'Bradburton'):
            return 3
        elif (word == 'freestyler'):
            return 3
        elif (word == 'Wikipedia'):
            return 5
        elif (word == 'stupider'):
            return 3
        elif (word == 'Franish'):
            return 2
        elif (word == "insomnia's"):
            return 4
        elif (word == 'neighbours'):
            return 2
        elif (word == "piper's"):
            return 2
        elif (word == 'Dundies'):
            return 2
        elif (word == 'Dunder'):
            return 2
        elif (word == 'Mifflen'):
            return 2
        elif (word == 'TMI'):
            return 3
        elif (word == 'offence'):
            return 2
        elif (word == 'tinfoil'):
            return 2
        elif (word == 'reciept'):
            return 2
        elif (word == 'shoulda'):
            return 2
        elif (word == 'overstimulated'):
            return 6
        elif (word == "Laker's"):
            return 2
        elif (word == "lighter's"):
            return 2
        elif (word == 'mm'):
            return 1
        elif (word == 'elevenths'):
            return 3
        elif (word == 'cath-lympics'):
            return 3
        elif (word == 'Cudi'):
            return 2
        elif (word == 'DDXing'):
            return 4
        elif (word == "'Nam"):
            return 1
        elif (word == 'counselling'):
            return 3
        elif (word == 'shirtless'):
            return 2
        elif (word == '\\imimportant'):
            return 4
        elif (word == 'pubescent'):
            return 3
        elif (word == 'MRI'):
            return 3
        else:
            print(word)
            return 2


# write the sentence-level features to file, computes syllables per sec while writing
def write_sentence(utterances, dataset):
    # writing sentence-level features to file
    os.chdir('sentenceFeatures/' + dataset)

    for (name, u) in utterances.items():
        # get syllable count from dictionary and sound data, calculate speaking rate ***** TO DO
        with open(name + '.txt', 'w') as f:
            if u.syllables != -1:
                f.write("Mean pitch: " + str(u.meanPitch) + " logHz\n" +
                        "Pitch standard deviation: " + str(u.pitchSD) + " logHz\n" +
                        "Pitch range: " + str(u.pRange) + " logHz\n" +
                        "Mean intensity: " + str(u.meanInten) + " dB\n" +
                        "Intensity range: " + str(u.iRange) + " dB\n" +
                        'Speaking Rate: ' + str(u.syllables) + ' syllables/sec')
                pass

    os.chdir('../..')


# normalize the bigrams based on the unigrams over entire corpus
def normalize(bigram, unigramTotals):
    for cols in bigram:
        if (unigramTotals[0] != 0):
            cols[0] = cols[0] / unigramTotals[0]
        if (unigramTotals[1] != 0):
            cols[1] = cols[1] / unigramTotals[1]
        if (unigramTotals[2] != 0):
            cols[2] = cols[2] / unigramTotals[2]


# convert 1st dataset's labels to binary
def to_binary(utterances, dataset):
    for u in utterances:
        if dataset == 'set1':
            if u.sarcasm == 3 or u.sarcasm == 1:
                u.sarcasm = 1
            elif u.sarcasm == 2 or u.sarcasm == 4:
                u.sarcasm = 0
        elif dataset == 'set2':
            if u.sarcasm == 3 or u.sarcasm == 2 or u.sarcasm == 1:
                u.sarcasm = 1
            elif u.sarcasm == 4:
                u.sarcasm = 0
        else:
            return
# sorting helper function
def soundDataSort(sd):
    return sd.name