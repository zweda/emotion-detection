import io
import re

TRAIN_FILE = './datasets/train.txt'
SOLUTION_FILE_EMOJI = './solutions/'

badWords = ['fuck', 'bitch', 'hate', 'kill', 'stupid']
repeatedChars = ['.', '?', '!']

emoticonsCount = {'happy':0, 'sad':0, 'angry':0, 'others':0}
badWordsCount = {'happy':0, 'sad':0, 'angry':0, 'others':0}
repeatedPunct = {'happy':0, 'sad':0, 'angry':0, 'others':0}

def countExamples():
    with io.open(TRAIN_FILE, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            lineSplit = line.strip().split('\t')
            label = lineSplit[4]
            emoticonsCount[label] += sum(1 for _ in re.finditer(r'[\U0001f600-\U0001f64f]', line))
            for w in badWords:
                badWordsCount[label] += sum(1 for _ in re.finditer(w, line))
            for c in repeatedChars:
                lineSplit = line.split(c)
                if len(lineSplit) > 1:
                    repeatedPunct[label] += 1

    print('Emoticon frequencies: ', emoticonsCount)
    print('Bad words count: ', badWordsCount)
    print('Repeated punctuations: ', repeatedPunct)

# Emoticon frequencies:  {'happy': 4719, 'sad': 3135, 'angry': 1216, 'others': 2016}
# Bad words count:  {'happy': 35, 'sad': 94, 'angry': 2187, 'others': 188}
# Repeated punctuations:  {'happy': 3558, 'sad': 4735, 'angry': 4840, 'others': 13272}

def logDifferentExamples():
    sol1 = './solutions/baseline-EP30_LR300e-5_LDim64_BS200-solution.txt'
    sol2 = './solutions/twitter-emoji-replaced-EP10_LR300e-5_LDim128_BS100-solution.txt'
    test = './datasets/test.txt'

    fin1 = open(sol1, encoding='utf8')
    fin1.readline()
    fin2 = open(sol2, encoding='utf8')
    fin2.readline()
    fin3 = open(test, encoding='utf8')
    fin3.readline()

    lines1 = []
    lines2 = []
    tests = []
    for l in fin1:
        lines1.append(l)
    for l in fin2:
        lines2.append(l)
    for l in fin3:
        tests.append(l)

    for i in range(len(lines1)):
        ls1 = lines1[i].strip().split('\t')
        ls2 = lines2[i].strip().split('\t')
        ls3 = tests[i].strip().split('\t')

        if ls1[4] != ls2[4] and ls2[4] == ls3[4] and ls2[4] != 'others':
            print("TRUE: ", tests[i])
            print("BASELINE: ", lines1[i])
            print("EMOJI: ", lines2[i])
            print()
            print()

    fin1.close()
    fin2.close()
    fin3.close()

def numberOfRepeatedPuncts():
    with io.open('./datasets/train.txt', encoding="utf8") as finput:
        finput.readline()
        repeated = 0
        for line in finput:
            repeatedChars = ['.', '?', '!']
            for c in repeatedChars:
                lineSplit = line.split(c)
                if len(lineSplit) > 1:
                    repeated+=1

        print("Number of repeated punctuation in train set: ", repeated)


if __name__ == '__main__':
    countExamples()
