"""
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


class ESIM(BaseModel):
    """
    别名：:class:`fastNLP.models.ESIM`  :class:`fastNLP.models.snli.ESIM`

    ESIM model的一个PyTorch实现
    论文参见： https://arxiv.org/pdf/1609.06038.pdf

    :param init_embedding: 初始化的Embedding
    :param int hidden_size: 隐藏层大小，默认值为Embedding的维度
    :param int num_labels: 目标标签种类数量，默认值为3
    :param float dropout_rate: dropout的比率，默认值为0.3
    :param float dropout_embed: 对Embedding的dropout比率，默认值为0.1
    """

    def __init__(self, init_embedding, hidden_size=None, num_labels=3, dropout_rate=0.3,
                 dropout_embed=0.1):
        super(ESIM, self).__init__()

        if isinstance(init_embedding, TokenEmbedding) or isinstance(init_embedding, Embedding):
            self.embedding = init_embedding
        else:
            self.embedding = Embedding(init_embedding)
        self.dropout_embed = EmbedDropout(p=dropout_embed)
        if hidden_size is None:
            hidden_size = self.embedding.embed_size
        self.rnn = BiRNN(self.embedding.embed_size, hidden_size, dropout_rate=dropout_rate)
        # self.rnn = LSTM(self.embedding.embed_size, hidden_size, dropout=dropout_rate, bidirectional=True)

        self.interfere = nn.Sequential(nn.Dropout(p=dropout_rate),
                                       nn.Linear(8 * hidden_size, hidden_size),
                                       nn.ReLU())
        nn.init.xavier_uniform_(self.interfere[1].weight.data)
        self.bi_attention = SoftmaxAttention()

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
        """
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
        lengths = x_mask.data.eq(1).long().sum(1)
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


def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = F.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    w_sum = weights.bmm(tensor)
    while mask.dim() < w_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(w_sum).contiguous().float()
    return w_sum * mask


class SoftmaxAttention(nn.Module):

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                              .contiguous())

        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                       .contiguous(),
                                       premise_mask)

        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses
