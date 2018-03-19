import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dataset



class CNN_text(nn.Module):
    def __init__(self, kernel_h=[3,4,5], kernel_num=100, embed_num=1000, embed_dim=300, dropout=0.5, L2_constrain=3, batchsize=50, pretrained_embeddings=None):
        super(CNN_text, self).__init__()

        self.embedding = nn.Embedding(embed_num,embed_dim)
        self.dropout = nn.Dropout(dropout)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        #the network structure
        #Conv2d: input- N,C,H,W output- (50,100,62,1)
        self.conv1 = nn.ModuleList([nn.Conv2d(1, 100, (K, 300)) for K in kernel_h])
        self.fc1 = nn.Linear(300,2)

    
    def max_pooling(self, x):
        x = F.relu(conv(x)).squeeze(3) #N,C,L - (50,100,62)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) 
        #x.size(2)=62  squeeze: (50,100,1) -> (50,100)
        return x
        

    def forward(self, x):
        x = self.embedding(x) #output: (N,H,W) = (50,64,300)
        x = x.unsqueeze(1) #(N,C,H,W)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1] #[N, C, H(50,100,62),(50,100,61),(50,100,60)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[N,C(50,100),(50,100),(50,100)]
        x = torch.cat(x,1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
