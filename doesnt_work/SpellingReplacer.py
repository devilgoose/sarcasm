import re
from textblob import TextBlob


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def split_string(string):
    match = re.match(r'(.*?)(?:\1)*$', string)
    word = match.group(1)
    return [word] * (len(string)//len(word))


def spell_correction(sentence):
    #This part splits repeating words
    new_sentence1 = ""
    for word in sentence.split():
        list_of_repeating_words = split_string(word)
        for w in list_of_repeating_words:
            new_sentence1 += str(w) + " "
    #Then this part deals with the extended words like loooong
    new_sentence2 = ""
    for word in new_sentence1.split():
        new_sentence2 += reduce_lengthening(word) + " "
    new_sentence2 = new_sentence2.rstrip()
    #Then we correct spelling
    #blob = TextBlob(new_sentence2)
    #new_sentence2 = str(blob.correct())
    return new_sentence2


sent = "I donnnnnt badbad goood speling"
nw = spell_correction(sent)
print(nw)
