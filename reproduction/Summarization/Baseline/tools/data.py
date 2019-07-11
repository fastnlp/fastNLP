#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import os
import re
import glob
import copy
import random
import json
import collections
from itertools import combinations
import numpy as np
from random import shuffle

import torch.utils.data
import time
import pickle

from nltk.tokenize import sent_tokenize

import tools.utils
from tools.logger import *

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
        :param vocab_file: string; path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
        :param max_size: int; The maximum size of the resulting Vocabulary.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, UNKNOWN_TOKEN,  START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf8') as vocab_f: #New : add the utf8 encoding to prevent error
            cnt = 0
            for line in vocab_f:
                cnt += 1
                pieces = line.split("\t")
                # pieces = line.split()
                w = pieces[0]
                # print(w)
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    logger.error('Duplicated word in vocabulary file Line %d : %s' % (cnt, w))
                    continue
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    logger.info("[INFO] max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        logger.info("[INFO] Finished constructing vocabulary of %i total words. Last word added: %s", self._count, self._id_to_word[self._count-1])

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def word_list(self):
        """Return the word list of the vocabulary"""
        return self._word_to_id.keys()

class Word_Embedding(object):
    def __init__(self, path, vocab):
        """
        :param path: string; the path of word embedding
        :param vocab: object;
        """
        logger.info("[INFO] Loading external word embedding...")
        self._path = path
        self._vocablist = vocab.word_list()
        self._vocab = vocab

    def load_my_vecs(self, k=200):
        """Load word embedding"""
        word_vecs = {}
        with open(self._path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                count += 1
                if word in self._vocablist:  # whether to judge if in vocab
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknown_words_by_zero(self, word_vecs, k=200):
        """Solve unknown by zeros"""
        zero = [0.0] * k
        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec

    def add_unknown_words_by_avg(self, word_vecs, k=200):
        """Solve unknown by avg word embedding"""
        # solve unknown words inplaced by zero list
        word_vecs_numpy = []
        for word in self._vocablist:
            if word in word_vecs:
                word_vecs_numpy.append(word_vecs[word])
        col = []
        for i in range(k):
            sum = 0.0
            for j in range(int(len(word_vecs_numpy))):
                sum += word_vecs_numpy[j][i]
                sum = round(sum, 6)
            col.append(sum)
        zero = []
        for m in range(k):
            avg = col[m] / int(len(word_vecs_numpy))
            avg = round(avg, 6)
            zero.append(float(avg))

        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        logger.info("[INFO] External Word Embedding iov count: %d, oov count: %d", iov, oov)
        return list_word2vec

    def add_unknown_words_by_uniform(self, word_vecs, uniform=0.25, k=200):
        """Solve unknown word by uniform(-0.25,0.25)"""
        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-1 * uniform, uniform, k).round(6).tolist()
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec

    # load word embedding
    def load_my_vecs_freq1(self, freqs, pro):
        word_vecs = {}
        with open(self._path, encoding="utf-8") as f:
            freq = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                if word in self._vocablist:  # whehter to judge if in vocab
                    if freqs[word] == 1:
                        a = np.random.uniform(0, 1, 1).round(2)
                        if pro < a:
                            continue
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

class DomainDict(object):
    """Domain embedding for Newsroom"""
    def __init__(self, path):
        self.domain_list = self.readDomainlist(path)
        # self.domain_list = ["foxnews.com", "cnn.com", "mashable.com", "nytimes.com", "washingtonpost.com"]
        self.domain_number = len(self.domain_list)
        self._domain_to_id = {}
        self._id_to_domain = {}
        self._cnt = 0

        self._domain_to_id["X"] = self._cnt
        self._id_to_domain[self._cnt] = "X"
        self._cnt += 1

        for i in range(self.domain_number):
            domain = self.domain_list[i]
            self._domain_to_id[domain] = self._cnt
            self._id_to_domain[self._cnt] = domain
            self._cnt += 1

    def readDomainlist(self, path):
        domain_list = []
        with open(path) as f:
            for line in f:
                domain_list.append(line.split("\t")[0].strip())
        logger.info(domain_list)
        return domain_list

    def domain2id(self, domain):
        """ Returns the id (integer) of a domain (string). Returns "X" for unknow domain.
        :param domain: string
        :return: id; int
        """
        if domain in self.domain_list:
            return self._domain_to_id[domain]
        else:
            logger.info(domain)
        return self._domain_to_id["X"]

    def id2domain(self, domain_id):
        """ Returns the domain (string) corresponding to an id (integer).
        :param id: int;
        :return: domain: string
        """
        if domain_id not in self._id_to_domain:
            raise ValueError('Id not found in DomainDict: %d' % domain_id)
        return self._id_to_domain[id]

    def size(self):
        return self._cnt


class Example(object):
    """Class representing a train/val/test example for text summarization."""
    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label, domainid=None):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.
            
        :param article_sents: list of strings; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list of strings; one per abstract sentence. In each sentence, each token is separated by a single space.
        :param domainid: int; publication of the example
        :param vocab: Vocabulary object
        :param sent_max_len: int; the maximum length of each sentence, padding all sentences to this length
        :param label: list of int; the index of selected sentences
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # origin_cnt = len(article_sents)
        # article_sents = [re.sub(r"\n+\t+", " ", sent) for sent in article_sents]
        # assert origin_cnt == len(article_sents)

        # Process the article
        for sent in article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words)) # store the length after truncation but before padding
            # self.enc_sent_input.append([vocab.word2id(w) for w in article_words]) # list of word ids; OOVs are represented by the id for UNK token
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words]) # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Store the original strings
        self.original_article = " ".join(article_sents)
        self.original_article_sents = article_sents

        if isinstance(abstract_sents[0], list):
            logger.debug("[INFO] Multi Reference summaries!")
            self.original_abstract_sents = []
            self.original_abstract = []
            for summary in abstract_sents:
                self.original_abstract_sents.append([sent.strip() for sent in summary])
                self.original_abstract.append("\n".join([sent.replace("\n", "") for sent in summary]))
        else:
            self.original_abstract_sents = [sent.replace("\n", "") for sent in abstract_sents]
            self.original_abstract = "\n".join(self.original_abstract_sents)

        # Store the label
        self.label = np.zeros(len(article_sents), dtype=int)
        if label != []:
            self.label[np.array(label)] = 1
        self.label = list(self.label)

        # Store the publication
        if domainid != None:
            if domainid == 0:
                logger.debug("domain id = 0!")
            self.domain = domainid

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i]
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            while len(article_words) < max_len:
                article_words.append(pad_id)
            self.enc_sent_input_pad.append(article_words)

class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) """
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, domaindict=None, randomX=False, usetag=False):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param domaindict: object; the domain dict to embed domain
        """
        self.domaindict = domaindict
        if domaindict:
            logger.info("[INFO] Use domain information in the dateset!")
            if randomX==True:
                logger.info("[INFO] Random some example to unknow domain X!")
                self.randomP = 0.1
        logger.info("[INFO] Start reading ExampleSet")
        start = time.time()
        self.example_list = []
        self.doc_max_timesteps = doc_max_timesteps
        cnt = 0
        with open(data_path, 'r') as reader:
            for line in reader:
                try:
                    e = json.loads(line)
                    article_sent = e['text']
                    tag = e["tag"][0] if usetag else e['publication']
                    # logger.info(tag)
                    if "duc" in data_path:
                        abstract_sent = e["summaryList"] if "summaryList" in e.keys() else [e['summary']]
                    else:
                        abstract_sent = e['summary']
                    if domaindict:
                        if randomX == True:
                            p = np.random.rand()
                            if p <= self.randomP:
                                domainid = domaindict.domain2id("X")
                            else:
                                domainid = domaindict.domain2id(tag)
                        else:
                            domainid = domaindict.domain2id(tag)
                    else:
                        domainid = None
                    logger.debug((tag, domainid))
                except (ValueError,EOFError) as e :
                    logger.debug(e)
                    break
                else:
                    example = Example(article_sent, abstract_sent, vocab, sent_max_len, e["label"], domainid) # Process into an Example.
                    self.example_list.append(example)
                cnt += 1
                # print(cnt)
            logger.info("[INFO] Finish reading ExampleSet. Total time is %f, Total size is %d", time.time() - start, len(self.example_list))
        self.size = len(self.example_list)

        # self.example_list.sort(key=lambda ex: ex.domain)

    def get_example(self, index):
        return self.example_list[index]

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            input_pad: [N, seq_len]
            label: [N]
            input_mask: [N]
            domain: [1]
        """
        item = self.example_list[index]
        input = np.array(item.enc_sent_input_pad)
        label = np.array(item.label, dtype=int)
        # pad input to doc_max_timesteps
        if len(input) < self.doc_max_timesteps:
            pad_number = self.doc_max_timesteps - len(input)
            pad_matrix = np.zeros((pad_number, len(input[0])))
            input_pad = np.vstack((input, pad_matrix))
            label = np.append(label, np.zeros(pad_number, dtype=int))
            input_mask = np.append(np.ones(len(input)), np.zeros(pad_number))
        else:
            input_pad = input[:self.doc_max_timesteps]
            label = label[:self.doc_max_timesteps]
            input_mask = np.ones(self.doc_max_timesteps)
        if self.domaindict:
            return torch.from_numpy(input_pad).long(), torch.from_numpy(label).long(), torch.from_numpy(input_mask).long(), item.domain
        return torch.from_numpy(input_pad).long(), torch.from_numpy(label).long(), torch.from_numpy(input_mask).long()

    def __len__(self):
        return self.size

class MultiExampleSet():
    def __init__(self, data_dir, vocab, doc_max_timesteps, sent_max_len, domaindict=None, randomX=False, usetag=False):
        self.datasets = [None] * (domaindict.size() - 1)
        data_path_list = [os.path.join(data_dir, s) for s in os.listdir(data_dir) if s.endswith("label.jsonl")]
        for data_path in data_path_list:
            fname = data_path.split("/")[-1]    # cnn.com.label.json
            dataname = ".".join(fname.split(".")[:-2])
            domainid = domaindict.domain2id(dataname)
            logger.info("[INFO] domain name: %s, domain id: %d" % (dataname, domainid))
            self.datasets[domainid - 1] = ExampleSet(data_path, vocab, doc_max_timesteps, sent_max_len, domaindict, randomX, usetag)

    def get(self, id):
        return self.datasets[id]

from torch.utils.data.dataloader import default_collate
def my_collate_fn(batch):
    '''
    :param batch: (input_pad, label, input_mask, domain)
    :return: 
    '''
    start_domain = batch[0][-1]
    # for i in range(len(batch)):
    #     print(batch[i][-1], end=',')
    batch = list(filter(lambda x: x[-1] == start_domain, batch))
    print("start_domain %d" % start_domain)
    print("batch_len %d" % len(batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据

