import numpy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class HAN(nn.Module):
    def __init__(self, input_size, output_size, 
                word_hidden_size, word_num_layers, word_context_size, 
                sent_hidden_size, sent_num_layers, sent_context_size):
        super(HAN, self).__init__()

        self.word_layer = AttentionNet(input_size, 
                                    word_hidden_size, 
                                    word_num_layers, 
                                    word_context_size)
        self.sent_layer = AttentionNet(2* word_hidden_size, 
                                    sent_hidden_size, 
                                    sent_num_layers, 
                                    sent_context_size)
        self.output_layer = nn.Linear(2* sent_hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch_doc):
        # input is a sequence of vector
        # if level == w, a seq of words (a sent); level == s, a seq of sents (a doc)
        doc_vec_list = []
        for doc in batch_doc:
            s_list = []
            for sent in doc:
                s_list.append(self.word_layer(sent))
            s_vec = torch.cat(s_list, dim=0)
            vec = self.sent_layer(s_vec)
            doc_vec_list.append(vec)
        doc_vec = torch.cat(doc_vec_list, dim=0)
        output = self.softmax(self.output_layer(doc_vec))
        return output

class AttentionNet(nn.Module):
    def __init__(self, input_size, gru_hidden_size, gru_num_layers, context_vec_size):
        super(AttentionNet, self).__init__()
        
        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.context_vec_size = context_vec_size

        # Encoder
        self.gru = nn.GRU(input_size=input_size, 
                        hidden_size=gru_hidden_size, 
                        num_layers=gru_num_layers, 
                        batch_first=False, 
                        bidirectional=True)
        # Attention
        self.fc = nn.Linear(2* gru_hidden_size, context_vec_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        # context vector
        self.context_vec = nn.Parameter(torch.Tensor(context_vec_size, 1))
        self.context_vec.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
        # inputs's dim (seq_len, word_dim)
        inputs = torch.unsqueeze(inputs, 1)
        h_t, hidden = self.gru(inputs)
        h_t = torch.squeeze(h_t, 1)
        u = self.tanh(self.fc(h_t))
        alpha = self.softmax(torch.mm(u, self.context_vec))
        output = torch.mm(h_t.t(), alpha).t()
        # output's dim (1, 2*hidden_size)
        return output


