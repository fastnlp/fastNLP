import math

import torch
from torch import nn

from fastNLP.modules.utils import mask_softmax


class Attention(torch.nn.Module):
    def __init__(self, normalize=False):
        super(Attention, self).__init__()
        self.normalize = normalize

    def forward(self, query, memory, mask):
        similarities = self._atten_forward(query, memory)
        if self.normalize:
            return mask_softmax(similarities, mask)
        return similarities

    def _atten_forward(self, query, memory):
        raise NotImplementedError


class DotAtte(nn.Module):
    def __init__(self, key_size, value_size):
        super(DotAtte, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)

    def forward(self, Q, K, V, seq_mask=None):
        """

        :param Q: [batch, seq_len, key_size]
        :param K: [batch, seq_len, key_size]
        :param V: [batch, seq_len, value_size]
        :param seq_mask: [batch, seq_len]
        """
        output = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        if seq_mask is not None:
            output.masked_fill_(seq_mask.lt(1), -float('inf'))
        output = nn.functional.softmax(output, dim=2)
        return torch.matmul(output, V)


class MultiHeadAtte(nn.Module):
    def __init__(self, input_size, output_size, key_size, value_size, num_atte):
        super(MultiHeadAtte, self).__init__()
        self.in_linear = nn.ModuleList()
        for i in range(num_atte * 3):
            out_feat = key_size if (i % 3) != 2 else value_size
            self.in_linear.append(nn.Linear(input_size, out_feat))
        self.attes = nn.ModuleList([DotAtte(key_size, value_size) for _ in range(num_atte)])
        self.out_linear = nn.Linear(value_size * num_atte, output_size)

    def forward(self, Q, K, V, seq_mask=None):
        heads = []
        for i in range(len(self.attes)):
            j = i * 3
            qi, ki, vi = self.in_linear[j](Q), self.in_linear[j+1](K), self.in_linear[j+2](V)
            headi = self.attes[i](qi, ki, vi, seq_mask)
            heads.append(headi)
        output = torch.cat(heads, dim=2)
        return self.out_linear(output)
