import pickle
import json
import nltk
from nltk.tokenize import stanford

# f = open('dataset/review.json', encoding='utf-8')
# samples = []
# j = 0
# for i, line in enumerate(f.readlines()):
#     review = json.loads(line)
#     samples.append((review['stars'], review['text']))
#     if (i+1) % 5000 == 0:
#         print(i)
#         pickle.dump(samples, open('review/samples%d.pkl'%j, 'wb'))
#         j += 1
#         samples = []
# pickle.dump(samples, open('review/samples%d.pkl'%j, 'wb'))
samples = pickle.load(open('review/samples0.pkl', 'rb'))
# print(samples[0])

import os
os.environ['JAVAHOME'] = 'D:\\java\\bin\\java.exe'
path_to_jar = 'E:\\College\\fudanNLP\\stanford-corenlp-full-2018-02-27\\stanford-corenlp-3.9.1.jar'
tokenizer = stanford.CoreNLPTokenizer()

dirname = 'review'
dirname1 = 'reviews'

for fn in os.listdir(dirname):
    print(fn)
    precessed = []
    for stars, text in pickle.load(open(os.path.join(dirname, fn), 'rb')):
        tokens = []
        sents = nltk.tokenize.sent_tokenize(text)
        for s in sents:
            tokens.append(tokenizer.tokenize(s))
        precessed.append((stars, tokens))
        # print(tokens)
        if len(precessed) % 100 == 0:
            print(len(precessed))
    pickle.dump(precessed, open(os.path.join(dirname1, fn), 'wb'))
    
