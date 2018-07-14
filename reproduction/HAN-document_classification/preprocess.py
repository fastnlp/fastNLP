''''
    Tokenize yelp dataset's documents using stanford core nlp
'''

import json
import os
import pickle

import nltk
from nltk.tokenize import stanford

input_filename = 'review.json'

# config for stanford core nlp
os.environ['JAVAHOME'] = 'D:\\java\\bin\\java.exe'
path_to_jar = 'E:\\College\\fudanNLP\\stanford-corenlp-full-2018-02-27\\stanford-corenlp-3.9.1.jar'
tokenizer = stanford.CoreNLPTokenizer()

in_dirname = 'review'
out_dirname = 'reviews'

f = open(input_filename, encoding='utf-8')
samples = []
j = 0
for i, line in enumerate(f.readlines()):
    review = json.loads(line)
    samples.append((review['stars'], review['text']))
    if (i + 1) % 5000 == 0:
        print(i)
        pickle.dump(samples, open(in_dirname + '/samples%d.pkl' % j, 'wb'))
        j += 1
        samples = []
pickle.dump(samples, open(in_dirname + '/samples%d.pkl' % j, 'wb'))
# samples = pickle.load(open(out_dirname + '/samples0.pkl', 'rb'))
# print(samples[0])


for fn in os.listdir(in_dirname):
    print(fn)
    precessed = []
    for stars, text in pickle.load(open(os.path.join(in_dirname, fn), 'rb')):
        tokens = []
        sents = nltk.tokenize.sent_tokenize(text)
        for s in sents:
            tokens.append(tokenizer.tokenize(s))
        precessed.append((stars, tokens))
        # print(tokens)
        if len(precessed) % 100 == 0:
            print(len(precessed))
    pickle.dump(precessed, open(os.path.join(out_dirname, fn), 'wb'))
