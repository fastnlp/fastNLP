import os
import pickle

import nltk
import numpy as np
import torch

from model import *

class SentIter:
    def __init__(self, dirname, count):
        self.dirname = dirname
        self.count = int(count)

    def __iter__(self):
        for f in os.listdir(self.dirname)[:self.count]:
            with open(os.path.join(self.dirname, f), 'rb') as f:
                for y, x in pickle.load(f):
                    for sent in x:
                        yield sent

def train_word_vec():
    # load data
    dirname = 'reviews'
    sents = SentIter(dirname, 238)
    # define model and train
    model = models.Word2Vec(size=200, sg=0, workers=4, min_count=5)
    model.build_vocab(sents)
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
    def __init__(self, dirname, start_file, num_files, embedding):
        self.dirname = dirname
        self.num_files = num_files
        self._files = os.listdir(dirname)[start_file:start_file + num_files]
        self.embedding = embedding
        self._cache = [(-1, None) for i in range(5)]

    def get_doc(self, n):
        file_id = n // 5000
        idx = file_id % 5
        if self._cache[idx][0] != file_id:
            with open(os.path.join(self.dirname, self._files[file_id]), 'rb') as f:
                self._cache[idx] = (file_id, pickle.load(f))
        y, x = self._cache[idx][1][n % 5000]
        sents = []
        for s_list in x:
            sents.append(' '.join(s_list))
        x = '\n'.join(sents)
        return x, y-1

    def __len__(self):
        return len(self._files)*5000

    def __getitem__(self, n):
        file_id = n // 5000
        idx = file_id % 5
        if self._cache[idx][0] != file_id:
            print('load {} to {}'.format(file_id, idx))
            with open(os.path.join(self.dirname, self._files[file_id]), 'rb') as f:
                self._cache[idx] = (file_id, pickle.load(f))
        y, x = self._cache[idx][1][n % 5000]
        doc = []
        for sent in x:
            if len(sent) == 0:
                continue
            sent_vec = []
            for word in sent:
                vec = self.embedding.get_vec(word)
                sent_vec.append(vec.tolist())
            sent_vec = torch.Tensor(sent_vec)
            doc.append(sent_vec)
        if len(doc) == 0:
            doc = [torch.zeros(1,200)]
        return doc, y-1

def collate(iterable):
    y_list = []
    x_list = []
    for x, y in iterable:
        y_list.append(y)
        x_list.append(x)
    return x_list, torch.LongTensor(y_list)

def train(net, dataset, num_epoch, batch_size, print_size=10, use_cuda=False):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=collate,
                            num_workers=0)
    running_loss = 0.0

    if use_cuda:
        net.cuda()
    print('start training')
    for epoch in range(num_epoch):
        for i, batch_samples in enumerate(dataloader):
            x, y = batch_samples
            doc_list = []
            for sample in x:
                doc = []
                for sent_vec in sample:
                    if use_cuda:
                        sent_vec = sent_vec.cuda()
                    doc.append(Variable(sent_vec))
                doc_list.append(pack_sequence(doc))
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

    train_word_vec()

    embed_model = Word2Vec.load('yelp.word2vec')
    embedding = Embedding_layer(embed_model.wv, embed_model.wv.vector_size)
    del embed_model
    start_file = 0
    dataset = YelpDocSet('reviews', start_file, 120-start_file, embedding)
    print('training data size {}'.format(len(dataset)))
    net = HAN(input_size=200, output_size=5, 
            word_hidden_size=50, word_num_layers=1, word_context_size=100,
            sent_hidden_size=50, sent_num_layers=1, sent_context_size=100)
    try:
        net.load_state_dict(torch.load('model.dict'))
        print("last time trained model has loaded")
    except Exception:
        print("cannot load model, train the inital model")
    
    train(net, dataset, num_epoch=5, batch_size=64, use_cuda=True)