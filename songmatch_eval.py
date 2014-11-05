import json
import re
import string
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
from nltk.metrics.scores import f_measure, accuracy
from nltk.metrics.confusionmatrix import ConfusionMatrix

regex = re.compile('[%s]' % re.escape(string.punctuation))

def preproc(s):
    s = s.lower()
    return s

def is_match(metric):
    if (metric < 0.7):
        return True
    return False

json_file = open('data/Bohemian-Rhapsody_train.json')
json_response = json.load(json_file)
json_file.close()

# Features
top_entry = json_response[0]
true_matches = [bool(song['Match']) for song in json_response[1:]]

FEATURE = 'SongName'
NGRAMS = 2
top_entry_value = preproc(top_entry[FEATURE])
print 'Comparing song name to top match reference:', top_entry[FEATURE]
top_entry_word_bigrams = set(ngrams(word_tokenize(top_entry_value), NGRAMS))

matches = []
for song in json_response[1:]:

    this_value = preproc(song[FEATURE])
    print '\t%s' % song[FEATURE]

    this_word_bigrams = set(ngrams(word_tokenize(this_value), NGRAMS))
    wbg_distance = jaccard_distance(top_entry_word_bigrams, this_word_bigrams)
    print '\t\tWord bigrams + Jaccard:\t'+str(wbg_distance)

    is_this_match = is_match(wbg_distance)
    print '\t\tMatch?', is_this_match
    matches.append(is_this_match)

cm = ConfusionMatrix(true_matches, matches)
print 'Confusion matrix'
print cm

print 'Accuracy:', accuracy(true_matches, matches)
