import pickle
import random

import torch
from torch.autograd import Variable


def float_wrapper(x, requires_grad=True, using_cuda=True):
    """
    transform float type list to pytorch variable
    """
    if using_cuda==True:
        return Variable(torch.FloatTensor(x).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(x), requires_grad=requires_grad)

def long_wrapper(x, requires_grad=True, using_cuda=True):
    """
    transform long type list to pytorch variable
    """
    if using_cuda==True:
        return Variable(torch.LongTensor(x).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.LongTensor(x), requires_grad=requires_grad)
    
def pad(X, using_cuda):
        """
        zero-pad sequnces to same length then pack them together
        """
        maxlen = max([x.size(0) for x in X])
        Y = []
        for x in X:
            padlen = maxlen - x.size(0)
            if padlen > 0:
                if using_cuda:
                    paddings = Variable(torch.zeros(padlen).long()).cuda()
                else:
                    paddings = Variable(torch.zeros(padlen).long())
                x_ = torch.cat((x, paddings), 0)
                Y.append(x_)
            else:
                Y.append(x)
        return torch.stack(Y)

class DataLoader(object):
    """
    load data with form {"feature", "class"}

    Args:
    fdir : data file address
    batch_size : batch_size
    shuffle : if True, shuffle dataset every epoch
    using_cuda : if True, return tensors on GPU
    """
    def __init__(self, fdir, batch_size, shuffle=True, using_cuda=True):
        with open(fdir, "rb") as f:
            self.data = pickle.load(f)
        self.batch_size = batch_size
        self.num = len(self.data)
        self.count = 0
        self.iters = int(self.num / batch_size)
        self.shuffle = shuffle
        self.using_cuda = using_cuda
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.count == self.iters:
            self.count = 0
            if self.shuffle:
                random.shuffle(self.data)
            raise StopIteration()
        else:
            batch = self.data[self.count * self.batch_size : (self.count + 1) * self.batch_size]
            self.count += 1
            X = [long_wrapper(x["sent"], using_cuda=self.using_cuda, requires_grad=False) for x in batch]
            X = pad(X, self.using_cuda)
            y = long_wrapper([x["class"] for x in batch], using_cuda=self.using_cuda, requires_grad=False)
            return {"feature" : X, "class" : y}
            

