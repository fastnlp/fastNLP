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
                 rnn_num_layer,
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
        self.layers = rnn_num_layer
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
        """
        :param x: LongTensor, [batch_size, mex_len]
        :return y: [batch_size, tag_size, tag_size]
        """
        x = self.Emb(x)
        # [batch_size, max_len, word_emb_dim]
        x, hidden = self.rnn(x)
        # [batch_size, max_len, hidden_size * direction]
        y = self.linear(x)
        # [batch_size, max_len, num_classes]
        return y

    def loss(self, x, y, mask, batch_size, max_len):
        """
        Negative log likelihood loss.
        :param x: FloatTensor, [batch_size, tag_size, tag_size]
        :param y: LongTensor, [batch_size, max_len]
        :param mask: ByteTensor, [batch_size, max_len]
        :param batch_size: int
        :param max_len: int
        :return loss:
                prediction:
        """
        x = x.float()
        y = y.long()
        mask = mask.byte()
        # print(x.shape, y.shape, mask.shape)

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
