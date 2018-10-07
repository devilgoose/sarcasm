import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('stopwords')
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))


def pos_neg_replacer(sentence):
    new_sentence = ""
    for word in sentence.split():
        print(word)
        senti = swn.senti_synsets(word, 'a')
        try:
            senti0 = list(senti)[0]
        except:
            new_sentence = new_sentence + word + " "
            continue
        pos = senti0.pos_score()
        neg = senti0.neg_score()
        print(pos)
        print(neg)
        if pos > 0.7:
            new_sentence = new_sentence + "positive "
        elif neg > 0.7:
            new_sentence = new_sentence + "negative "
        else:
            new_sentence = new_sentence + word + " "
    new_sentence = new_sentence.rstrip()
    return new_sentence


word_up = "happy bad sad about okay we are am among worst terrible"
print(word_up)
gang = pos_neg_replacer(word_up)
print(gang)