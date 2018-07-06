import torch
import torch.nn as nn
from torch.nn import functional as F

from fastNLP.models.base_model import BaseModel
from fastNLP.modules.CRF import ContionalRandomField


class SeqLabeling(BaseModel):
    """
    PyTorch Network for sequence labeling
    """

    def __init__(self, hidden_dim,
                 rnn_num_layerd,
                 num_classes,
                 vocab_size,
                 word_emb_dim=100,
                 init_emb=None,
                 rnn_mode="gru",
                 bi_direction=False,
                 dropout=0.5,
                 use_crf=True):
        super(SeqLabeling, self).__init__()

        self.Emb = nn.Embedding(vocab_size, word_emb_dim)
        if init_emb:
            self.Emb.weight = nn.Parameter(init_emb)

        self.num_classes = num_classes
        self.input_dim = word_emb_dim
        self.layers = rnn_num_layerd
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction
        self.dropout = dropout
        self.mode = rnn_mode

        if self.mode == "lstm":
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                               bidirectional=self.bi_direction, dropout=self.dropout)
        elif self.mode == "gru":
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                              bidirectional=self.bi_direction, dropout=self.dropout)
        elif self.mode == "rnn":
            self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                              bidirectional=self.bi_direction, dropout=self.dropout)
        else:
            raise Exception
        if bi_direction:
            self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)
        else:
            self.linear = nn.Linear(self.hidden_dim, self.num_classes)
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = ContionalRandomField(num_classes)

    def forward(self, x):

        x = self.embedding(x)
        x, hidden = self.encode(x)
        x = self.aggregation(x)
        x = self.output(x)
        return x

    def embedding(self, x):
        return self.Emb(x)

    def encode(self, x):
        return self.rnn(x)

    def aggregate(self, x):
        return x

    def decode(self, x):
        x = self.linear(x)
        return x

    def loss(self, x, y, mask, batch_size, max_len):
        """
        Negative log likelihood loss.
        :param x:
        :param y:
        :param seq_len:
        :return loss:
                prediction:
        """
        if self.use_crf:
            total_loss = self.crf(x, y, mask)
            tag_seq = self.crf.viterbi_decode(x, mask)
        else:
            # error
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            x = x.view(batch_size * max_len, -1)
            score = F.log_softmax(x)
            total_loss = loss_function(score, y.view(batch_size * max_len))
            _, tag_seq = torch.max(score)
            tag_seq = tag_seq.view(batch_size, max_len)
        return torch.mean(total_loss), tag_seq
