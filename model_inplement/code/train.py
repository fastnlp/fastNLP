import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import nltk
import numpy as np
import torch

from model import *

UNK_token = '/unk'
            
class SampleIter:
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        for f in os.listdir(self.dirname):
            for y, x in pickle.load(open(os.path.join(self.dirname, f), 'rb')):
                yield x, y

class SentIter:
    def __init__(self, dirname, count, vocab=None):
        self.dirname = dirname
        self.count = int(count)
        self.vocab = None

    def __iter__(self):
        for f in os.listdir(self.dirname)[:self.count]:
            for y, x in pickle.load(open(os.path.join(self.dirname, f), 'rb')):
                for sent in x:
                    if self.vocab is not None:
                        _sent = []
                        for w in sent:
                            if w in self.vocab:
                                _sent.append(w)
                            else:
                                _sent.append(UNK_token)
                        sent = _sent
                    yield sent

def train_word_vec():
    # load data
    dirname = 'reviews'
    sents = SentIter(dirname, 238)
    # define model and train
    model = models.Word2Vec(size=200, sg=0, workers=4, min_count=5)
    model.build_vocab(sents)
    sents.vocab = model.wv.vocab
    model.train(sents, total_examples=model.corpus_count, epochs=10)
    model.save('yelp.word2vec')
    print(model.wv.similarity('woman', 'man'))
    print(model.wv.similarity('nice', 'awful'))

class Embedding_layer:
    def __init__(self, wv, vector_size):
        self.wv = wv
        self.vector_size = vector_size

    def get_vec(self, w):
        try:
            v = self.wv[w]
        except KeyError as e:
            v = np.random.randn(self.vector_size)
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
        doc = []
        for sent in x:
            sent_vec = []
            for word in sent:
                vec = self.embedding.get_vec(word)
                vec = torch.Tensor(vec.reshape((1, -1)))
                sent_vec.append(vec)
            sent_vec = torch.cat(sent_vec, dim=0)
            # print(sent_vec.size())
            doc.append(sent_vec)
        return doc, y-1

def collate(iterable):
    y_list = []
    x_list = []
    for x, y in iterable:
        y_list.append(y)
        x_list.append(x)
    return x_list, torch.LongTensor(y_list)

def train(net, dataset, num_epoch, batch_size, print_size=10, use_cuda=False):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=collate,
                            num_workers=0)
    running_loss = 0.0

    if use_cuda:
        net.cuda()
    for epoch in range(num_epoch):
        for i, batch_samples in enumerate(dataloader):
            x, y = batch_samples
            doc_list = []
            for sample in x:
                doc = []
                for sent_vec in sample:
                    # print(sent_vec.size())
                    if use_cuda:
                        sent_vec = sent_vec.cuda()
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
                print('{}, {}'.format(i+1, running_loss/print_size))
                running_loss = 0.0
                torch.save(net.state_dict(), 'model.dict')
    torch.save(net.state_dict(), 'model.dict')
    

if __name__ == '__main__':
    '''
    Train process
    '''
    from gensim.models import Word2Vec
    import gensim
    from gensim import models

    # train_word_vec()

    embed_model = Word2Vec.load('yelp.word2vec')
    embedding = Embedding_layer(embed_model.wv, embed_model.wv.vector_size)
    del embed_model
    dataset = YelpDocSet('reviews', 120, embedding)

    net = HAN(input_size=200, output_size=5, 
            word_hidden_size=50, word_num_layers=1, word_context_size=100,
            sent_hidden_size=50, sent_num_layers=1, sent_context_size=100)
    # net.load_state_dict(torch.load('model.dict'))
    
    train(net, dataset, num_epoch=1, batch_size=64, use_cuda=True)
