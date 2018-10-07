from textblob import TextBlob
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('punkt')

def check_word_count(word, sentences, mega_sentence):
    check = TextBlob(mega_sentence)
    return check.word_counts[word]


def remove_one_off(sentences):
    output_sentences = []
    new_sentence = ""
    mega_sentence = ""
    for sentence in sentences:
        mega_sentence = mega_sentence + sentence + " "
    for sentence in sentences:
        new_sentence = ''
        for word in sentence.split():
            if not wordnet.synsets(word):
                count = check_word_count(word, sentences, mega_sentence)
                if count < 2:
                    continue
                else:
                    new_sentence = new_sentence + word + " "
            else:
                new_sentence = new_sentence + word + " "
        new_sentence = new_sentence.rstrip()
        output_sentences.append(new_sentence)
    return output_sentences
