import re
import sys
import itertools
import numpy as np
from torch.utils.data import Dataset, DataLoader

import random
import os
import pickle
import codecs
from gensim import corpora
import gensim


def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

def pad_sentences(sentence, padding_word=" <PAD/>"):
    sequence_length = 64
    sent = sentence.split()
    padded_sentence = sentence + padding_word * (sequence_length - len(sent))
    return padded_sentence



#data loader
class MRDataset(Dataset):
    def __init__(self):

        #load positive and negative sentenses from files
        with codecs.open("./rt-polaritydata/rt-polarity.pos",encoding ='ISO-8859-1') as f:
            positive_examples = list(f.readlines())
        with codecs.open("./rt-polaritydata/rt-polarity.neg",encoding ='ISO-8859-1') as f:
            negative_examples = list(f.readlines())
        #s.strip: clear "\n"; clear_str; pad
        positive_examples = [pad_sentences(clean_str(s.strip())) for s in positive_examples]
        negative_examples = [pad_sentences(clean_str(s.strip())) for s in negative_examples]
        self.examples = positive_examples + negative_examples
        self.sentences_texts = [sample.split() for sample in self.examples]

        #word dictionary
        dictionary = corpora.Dictionary(self.sentences_texts) 
        self.word2id_dict = dictionary.token2id  # transform to dict, like {"human":0, "a":1,...}

        #set lables: postive is 1; negative is 0
        positive_labels = [1 for _ in positive_examples]
        negative_labels = [0 for _ in negative_examples]
        self.lables = positive_labels + negative_labels
        examples_lables = list(zip(self.examples,self.lables))
        random.shuffle(examples_lables)
        self.MRDataset_frame = examples_lables

        #transform word to id
        self.MRDataset_wordid = \
            [(
                np.array([self.word2id_dict[word] for word in sent[0].split()], dtype=np.int64), 
                sent[1]
            ) for sent in self.MRDataset_frame]
        
    def word_embeddings(self, path = './GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'):
	    #establish from google
	    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
	    print('Please wait ... (it could take a while to load the file : {})'.format(path))

	    word_dict = self.word2id_dict
	    embedding_weights = np.random.uniform(-0.25, 0.25, (len(self.word2id_dict), 300))

	    for word in word_dict:
            word_id = word_dict[word]
            if word in model.wv.vocab:
                embedding_weights[word_id, :] = model[word]

	    return embedding_weights

    def __len__(self):

        return len(self.MRDataset_frame)

    def __getitem__(self,idx):

        sample = self.MRDataset_wordid[idx]      
        return sample

    def getsent(self, idx):

        sample = self.MRDataset_wordid[idx][0]       
        return sample

    def getlabel(self, idx):

        label = self.MRDataset_wordid[idx][1]
        return label


    def word2id(self):
        
        return self.word2id_dict

    def id2word(self):

        id2word_dict = dict([val,key] for key,val in self.word2id_dict.items()) 
        return id2word_dict
    

class train_set(Dataset):

    def __init__(self, samples):

        self.train_frame = samples

    def __len__(self):

        return len(self.train_frame)

    def __getitem__(self, idx):

        return self.train_frame[idx]


class test_set(Dataset):

    def __init__(self, samples):

        self.test_frame = samples

    def __len__(self):

        return len(self.test_frame)

    def __getitem__(self, idx):

        return self.test_frame[idx]
