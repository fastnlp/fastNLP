import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_model import BaseModel
from torch.autograd import Variable

USE_GPU = True


def to_var(x):
    if torch.cuda.is_available() and USE_GPU:
        x = x.cuda()
    return Variable(x)


class WordSegModel(BaseModel):
    """
        Model controller for WordSeg
    """

    def __init__(self):
        super(WordSegModel, self).__init__()
        self.id2word = None
        self.word2id = None
        self.id2tag = None
        self.tag2id = None

        self.lstm_batch_size = 8
        self.lstm_seq_len = 32  # Trainer batch_size == lstm_batch_size * lstm_seq_len
        self.hidden_dim = 100
        self.lstm_num_layers = 2
        self.vocab_size = 100
        self.word_emb_dim = 100

        self.model = WordSeg(self.hidden_dim, self.lstm_num_layers, self.vocab_size, self.word_emb_dim)
        self.hidden = (to_var(torch.zeros(2, self.lstm_batch_size, self.word_emb_dim)),
                       to_var(torch.zeros(2, self.lstm_batch_size, self.word_emb_dim)))

        self.optimizer = None
        self._loss = None

    def prepare_input(self, data):
        """
            perform word indices lookup to convert strings into indices
            :param data: list of string, each string contains word + space + [B, M, E, S]
            :return
        """
        word_list = []
        tag_list = []
        for line in data:
            if len(line) > 2:
                tokens = line.split("#")
                word_list.append(tokens[0])
                tag_list.append(tokens[2][0])
        self.id2word = list(set(word_list))
        self.word2id = {word: idx for idx, word in enumerate(self.id2word)}
        self.id2tag = list(set(tag_list))
        self.tag2id = {tag: idx for idx, tag in enumerate(self.id2tag)}
        words = np.array([self.word2id[w] for w in word_list]).reshape(-1, 1)
        tags = np.array([self.tag2id[t] for t in tag_list]).reshape(-1, 1)
        return words, tags

    def mode(self, test=False):
        if test:
            self.model.eval()
        else:
            self.model.train()

    def data_forward(self, x):
        """
        :param x: sequence of length [batch_size], word indices
        :return:
        """
        x = x.reshape(self.lstm_batch_size, self.lstm_seq_len)
        output, self.hidden = self.model(x, self.hidden)
        return output

    def define_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.85)

    def get_loss(self, pred, truth):

        self._loss = nn.CrossEntropyLoss(pred, truth)
        return self._loss

    def grad_backward(self):
        self.model.zero_grad()
        self._loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 5, norm_type=2)
        self.optimizer.step()


class WordSeg(nn.Module):
    """
        PyTorch Network for word segmentation
    """

    def __init__(self, hidden_dim, lstm_num_layers, vocab_size, word_emb_dim=100):
        super(WordSeg, self).__init__()

        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim
        self.lstm_num_layers = lstm_num_layers
        self.hidden_dim = hidden_dim

        self.word_emb = nn.Embedding(self.vocab_size, self.word_emb_dim)

        self.lstm = nn.LSTM(input_size=self.word_emb_dim,
                            hidden_size=self.word_emb_dim,
                            num_layers=self.lstm_num_layers,
                            bias=True,
                            dropout=0.5,
                            batch_first=True)

        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

    def forward(self, x, hidden):
        """
        :param x: tensor of shape [batch_size, seq_len], vocabulary index
        :param hidden:
        :return x: probability of vocabulary entries
                hidden: (memory cell, hidden state) from LSTM
        """
        # [batch_size, seq_len]
        x = self.word_emb(x)
        # [batch_size, seq_len, word_emb_size]
        x, hidden = self.lstm(x, hidden)
        # [batch_size, seq_len, word_emb_size]
        x = x.contiguous().view(x.shape[0] * x.shape[1], -1)
        # [batch_size*seq_len, word_emb_size]
        x = self.linear(x)
        # [batch_size*seq_len, vocab_size]
        return x, hidden
