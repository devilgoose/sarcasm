

# compile all features together into one .arff file
def compile_weka(samples, file_name):

    with open(file_name + '.arff', 'w') as f:
        f.write('''@RELATION "prosodic features"
    
    @ATTRIBUTE meanpitch numeric
    @ATTRIBUTE pitchrange numeric
    @ATTRIBUTE pitchsd numeric
    @ATTRIBUTE meanintensity numeric
    @ATTRIBUTE intensityrange numeric
    @ATTRIBUTE speakingrate numeric
    @ATTRIBUTE class {0,1}
    
    @DATA
    ''')

        for s in samples:
            del s[-1]
            f.write(str(s[0]))
            for i in range(1,len(s)):
                f.write(',' + str(s[i]))
            f.write('\n')


def compile_csv(samples, file_name):

    with open(file_name + '.csv', 'w') as f:
        for sample in samples:
            for feature in sample:
                f.write(str(feature) + ',')
            f.write('\n')