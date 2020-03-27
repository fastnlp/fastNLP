r"""
.. todo::
    doc
"""
__all__ = [
    "ESIM"
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .base_model import BaseModel
from ..core.const import Const
from ..core.utils import seq_len_to_mask
from ..embeddings.embedding import TokenEmbedding, Embedding
from ..modules.encoder import BiAttention


class ESIM(BaseModel):
    r"""
    ESIM model的一个PyTorch实现
    论文参见： https://arxiv.org/pdf/1609.06038.pdf

    """

    def __init__(self, embed, hidden_size=None, num_labels=3, dropout_rate=0.3,
                 dropout_embed=0.1):
        r"""
        
        :param embed: 初始化的Embedding
        :param int hidden_size: 隐藏层大小，默认值为Embedding的维度
        :param int num_labels: 目标标签种类数量，默认值为3
        :param float dropout_rate: dropout的比率，默认值为0.3
        :param float dropout_embed: 对Embedding的dropout比率，默认值为0.1
        """
        super(ESIM, self).__init__()

        if isinstance(embed, TokenEmbedding) or isinstance(embed, Embedding):
            self.embedding = embed
        else:
            self.embedding = Embedding(embed)
        self.dropout_embed = EmbedDropout(p=dropout_embed)
        if hidden_size is None:
            hidden_size = self.embedding.embed_size
        self.rnn = BiRNN(self.embedding.embed_size, hidden_size, dropout_rate=dropout_rate)
        # self.rnn = LSTM(self.embedding.embed_size, hidden_size, dropout=dropout_rate, bidirectional=True)

        self.interfere = nn.Sequential(nn.Dropout(p=dropout_rate),
                                       nn.Linear(8 * hidden_size, hidden_size),
                                       nn.ReLU())
        nn.init.xavier_uniform_(self.interfere[1].weight.data)
        self.bi_attention = BiAttention()

        self.rnn_high = BiRNN(self.embedding.embed_size, hidden_size, dropout_rate=dropout_rate)
        # self.rnn_high = LSTM(hidden_size, hidden_size, dropout=dropout_rate, bidirectional=True,)

        self.classifier = nn.Sequential(nn.Dropout(p=dropout_rate),
                                        nn.Linear(8 * hidden_size, hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(p=dropout_rate),
                                        nn.Linear(hidden_size, num_labels))

        self.dropout_rnn = nn.Dropout(p=dropout_rate)

        nn.init.xavier_uniform_(self.classifier[1].weight.data)
        nn.init.xavier_uniform_(self.classifier[4].weight.data)

    def forward(self, words1, words2, seq_len1, seq_len2, target=None):
        r"""
        :param words1: [batch, seq_len]
        :param words2: [batch, seq_len]
        :param seq_len1: [batch]
        :param seq_len2: [batch]
        :param target:
        :return:
        """
        mask1 = seq_len_to_mask(seq_len1, words1.size(1))
        mask2 = seq_len_to_mask(seq_len2, words2.size(1))
        a0 = self.embedding(words1)  # B * len * emb_dim
        b0 = self.embedding(words2)
        a0, b0 = self.dropout_embed(a0), self.dropout_embed(b0)
        a = self.rnn(a0, mask1.byte())  # a: [B, PL, 2 * H]
        b = self.rnn(b0, mask2.byte())
        # a = self.dropout_rnn(self.rnn(a0, seq_len1)[0])  # a: [B, PL, 2 * H]
        # b = self.dropout_rnn(self.rnn(b0, seq_len2)[0])

        ai, bi = self.bi_attention(a, mask1, b, mask2)

        a_ = torch.cat((a, ai, a - ai, a * ai), dim=2)  # ma: [B, PL, 8 * H]
        b_ = torch.cat((b, bi, b - bi, b * bi), dim=2)
        a_f = self.interfere(a_)
        b_f = self.interfere(b_)

        a_h = self.rnn_high(a_f, mask1.byte())  # ma: [B, PL, 2 * H]
        b_h = self.rnn_high(b_f, mask2.byte())
        # a_h = self.dropout_rnn(self.rnn_high(a_f, seq_len1)[0])  # ma: [B, PL, 2 * H]
        # b_h = self.dropout_rnn(self.rnn_high(b_f, seq_len2)[0])

        a_avg = self.mean_pooling(a_h, mask1, dim=1)
        a_max, _ = self.max_pooling(a_h, mask1, dim=1)
        b_avg = self.mean_pooling(b_h, mask2, dim=1)
        b_max, _ = self.max_pooling(b_h, mask2, dim=1)

        out = torch.cat((a_avg, a_max, b_avg, b_max), dim=1)  # v: [B, 8 * H]
        logits = torch.tanh(self.classifier(out))

        if target is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, target)

            return {Const.LOSS: loss, Const.OUTPUT: logits}
        else:
            return {Const.OUTPUT: logits}

    def predict(self, **kwargs):
        pred = self.forward(**kwargs)[Const.OUTPUT].argmax(-1)
        return {Const.OUTPUT: pred}

    # input [batch_size, len , hidden]
    # mask  [batch_size, len] (111...00)
    @staticmethod
    def mean_pooling(input, mask, dim=1):
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(input * masks, dim=dim) / torch.sum(masks, dim=1)

    @staticmethod
    def max_pooling(input, mask, dim=1):
        my_inf = 10e12
        masks = mask.view(mask.size(0), mask.size(1), -1)
        masks = masks.expand(-1, -1, input.size(2)).float()
        return torch.max(input + masks.le(0.5).float() * -my_inf, dim=dim)


class EmbedDropout(nn.Dropout):

    def forward(self, sequences_batch):
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(BiRNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers=1,
                           bidirectional=True,
                           batch_first=True)

    def forward(self, x, x_mask):
        # Sort x
        lengths = x_mask.data.eq(True).long().sum(1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])

        x = x.index_select(0, idx_sort)
        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        # Apply dropout to input
        if self.dropout_rate > 0:
            dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
            rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
        self.rnn.flatten_parameters()
        output = self.rnn(rnn_input)[0]
        # Unpack everything
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        output = output.index_select(0, idx_unsort)
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, padding], 1)
        return output

