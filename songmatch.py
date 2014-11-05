import os
import json
import re
import string
import requests
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance, jaccard_distance

regex = re.compile('[%s]' % re.escape(string.punctuation))

def preproc(s):
    s = s.lower()
    return s


DEBUG = True
if DEBUG:
    json_file = open('data/Bohemian-Rhapsody.json')
    json_response = json.load(json_file)
    json_file.close()
else:
    url = 'http://tinysong.com/s/Bohemian-Rhapsody?format=json&key=%s' % os.environ.get('TINYSONG_KEY')
    r = requests.get(url)
    json_response = r.json()

# Features
top_entry = json_response[0]

FEATURE = 'SongName'
NGRAMS = 2
top_entry_value = preproc(top_entry[FEATURE])
print 'Comparing song name to top match reference:', top_entry[FEATURE]
top_entry_char_bigrams = set(ngrams(top_entry_value, NGRAMS))
top_entry_word_bigrams = set(ngrams(word_tokenize(top_entry_value), NGRAMS))

for song in json_response[1:]:

    this_value = preproc(song[FEATURE])
    print '\t%s' % song[FEATURE]

    l_distance = edit_distance(top_entry_value, this_value)
    print '\t\tLevenshtein:\t\t'+str(l_distance)

    ld_distance = edit_distance(top_entry_value, this_value, True)
    print '\t\tLevenshtein-Damerau:\t'+str(ld_distance)

    this_char_bigrams = set(ngrams(this_value, NGRAMS))
    cbg_distance = jaccard_distance(top_entry_char_bigrams, this_char_bigrams)
    print '\t\tChar bigrams + Jaccard:\t'+str(cbg_distance)

    this_word_bigrams = set(ngrams(word_tokenize(this_value), NGRAMS))
    if len(this_word_bigrams) and len(top_entry_word_bigrams):
        wbg_distance = jaccard_distance(top_entry_word_bigrams, this_word_bigrams)
        print '\t\tWord bigrams + Jaccard:\t'+str(wbg_distance)
