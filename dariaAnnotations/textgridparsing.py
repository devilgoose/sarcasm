
import os
from os import walk

# parsing the praat textgrid files into something more readable by main data.py file

# container to store an annotation - has start time, end time, and the word
class boundary:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word



files = []

os.chdir('textgrid files')
current = os.getcwd()
for (_, _, filenames) in walk(os.getcwd()):
    files = filenames

files.sort()

for fi in files:
    utterances = []
    data = {}
    length = 0.0
    count = 0


    with open(fi, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.replace('"', '')
            line = line.split(' ')


            data[line[0]] = line[1:]
            
            if (line[0] == 'intervals' and count == 0):
                data = {}
                count += 1
            elif (line[0] == 'intervals'):

                start = data['xmin']
                end = data['xmax']
                word = data['text']
                if (word[-1] != ''):
                    annotation = boundary(float(start[-1]), float(end[-1]), word[-1])
                    utterances.append(annotation)
                data = {}
            elif (line[-1] == '[length]' and count != 0):
                start = data['xmin']
                end = data['xmax']
                word = data['text']
                if (word[-1] != ''):
                    annotation = boundary(float(start[-1]), float(end[-1]), word[-1])
                    utterances.append(annotation)
                break
                
                
        
    
    os.chdir('../parsedAnnotations')
    
    (name, _) = fi.split('.')

    with open(name + '.txt', 'w') as f:
        for a in utterances:
            f.writelines(str(a.start) + '\t' + str(a.end) + '\t' + str(a.word) + '\n')
    
    os.chdir(current)
