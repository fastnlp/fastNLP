import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import nltk
import numpy as np
import torch

from model import *
            
class SampleIter:
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        for f in os.listdir(self.dirname):
            for y, x in pickle.load(open(os.path.join(self.dirname, f), 'rb')):
                yield x, y

class SentIter:
    def __init__(self, dirname, count):
        self.dirname = dirname
        self.count = int(count)
    
    def __iter__(self):
        for f in os.listdir(self.dirname)[:self.count]:
            for y, x in pickle.load(open(os.path.join(self.dirname, f), 'rb')):
                for sent in x:
                    yield sent

def train_word_vec():
    # load data
    dirname = 'reviews'
    sents = SentIter(dirname, 238)
    # define model and train
    model = models.Word2Vec(sentences=sents, size=200, sg=0, workers=4, min_count=5)
    model.save('yelp.word2vec')

class Embedding_layer:
    def __init__(self, wv, vector_size):
        self.wv = wv
        self.vector_size = vector_size

    def get_vec(self, w):
        try:
            v = self.wv[w]
        except KeyError as e:
            v = np.zeros(self.vector_size)
        return v


from torch.utils.data import DataLoader, Dataset
class YelpDocSet(Dataset):
    def __init__(self, dirname, num_files, embedding):
        self.dirname = dirname
        self.num_files = num_files
        self._len = num_files*5000
        self._files = os.listdir(dirname)[:num_files]
        self.embedding = embedding
    
    def __len__(self):
        return self._len

    def __getitem__(self, n):
        file_id = n // 5000
        sample_list = pickle.load(open(
                os.path.join(self.dirname, self._files[file_id]), 'rb'))
        y, x = sample_list[n % 5000]
        return x, y-1

def collate(iterable):
    y_list = []
    x_list = []
    for x, y in iterable:
        y_list.append(y)
        x_list.append(x)
    return x_list, torch.LongTensor(y_list)

def train(net, num_epoch, batch_size, print_size=10, use_cuda=False):
    from gensim.models import Word2Vec
    import torch
    import gensim
    from gensim import models

    embed_model = Word2Vec.load('yelp.word2vec')
    embedding = Embedding_layer(embed_model.wv, embed_model.wv.vector_size)
    del embed_model

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    dirname = 'reviews'
    dataloader = DataLoader(YelpDocSet(dirname, 238, embedding), 
                            batch_size=batch_size, 
                            collate_fn=collate,
                            num_workers=4)
    running_loss = 0.0

    if use_cuda:
        net.cuda()
    for epoch in range(num_epoch):
        for i, batch_samples in enumerate(dataloader):
            x, y = batch_samples
            doc_list = []
            for sample in x:
                doc = []
                for sent in sample:
                    sent_vec = []
                    for word in sent:
                        vec = embedding.get_vec(word)
                        vec = torch.Tensor(vec.reshape((1, -1)))
                        if use_cuda:
                            vec = vec.cuda()
                        sent_vec.append(vec)
                    sent_vec = torch.cat(sent_vec, dim=0)
                    # print(sent_vec.size())
                    doc.append(Variable(sent_vec))
                doc_list.append(doc)
            if use_cuda:
                y = y.cuda()
            y = Variable(y)
            predict = net(doc_list)
            loss = criterion(predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if i % print_size == print_size-1:
                print(running_loss/print_size)
                running_loss = 0.0
                torch.save(net.state_dict(), 'model.dict')
    torch.save(net.state_dict(), 'model.dict')
    

if __name__ == '__main__':
    '''
    Train process
    '''
    

    net = HAN(input_size=200, output_size=5, 
            word_hidden_size=50, word_num_layers=1, word_context_size=100,
            sent_hidden_size=50, sent_num_layers=1, sent_context_size=100)
    
    train(net, num_epoch=1, batch_size=64, use_cuda=True)
